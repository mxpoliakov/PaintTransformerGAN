import torch
import numpy as np
from .base_model import BaseModel
from . import networks
from util import morphology
from scipy.optimize import linear_sum_assignment
import torchvision.transforms as transforms
from gan_stroke_generator.mypaint_gan_train_predict import GANMyPaintStrokes

class PainterModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='null')
        parser.add_argument('--used_strokes', type=int, default=8,
                            help='actually generated strokes number')
        parser.add_argument('--num_blocks', type=int, default=3,
                            help='number of transformer blocks for stroke generator')
        parser.add_argument('--lambda_w', type=float, default=10.0, help='weight for w loss of stroke shape')
        parser.add_argument('--lambda_pixel', type=float, default=10.0, help='weight for pixel-level L1 loss')
        parser.add_argument('--lambda_gt', type=float, default=1.0, help='weight for ground-truth loss')
        parser.add_argument('--lambda_decision', type=float, default=10.0, help='weight for stroke decision loss')
        parser.add_argument('--lambda_recall', type=float, default=10.0, help='weight of recall for stroke decision loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['pixel', 'gt', 'w', 'decision']
        self.visual_names = ['old', 'render', 'rec']
        self.model_names = ['g']
        self.d = 12
        self.d_shape = 9

        self.gan_strokes = GANMyPaintStrokes(action_size=self.d_shape)
        self.gan_strokes.load_from_train_checkpoint(
            '../gan_stroke_generator/gan_train_checkpoints/gan_mypaint_strokes_latest.tar'
        )
        net_g = networks.Painter(self.d_shape, opt.used_strokes, opt.ngf,
                                 n_enc_layers=opt.num_blocks, n_dec_layers=opt.num_blocks)
        self.net_g = networks.init_net(net_g, opt.init_type, opt.init_gain, self.gpu_ids)
        self.old = None
        self.render = None
        self.rec = None
        self.gt_param = None
        self.pred_param = None
        self.gt_decision = None
        self.pred_decision = None
        self.patch_size = 32
        self.loss_pixel = torch.tensor(0., device=self.device)
        self.loss_gt = torch.tensor(0., device=self.device)
        self.loss_w = torch.tensor(0., device=self.device)
        self.loss_decision = torch.tensor(0., device=self.device)
        self.criterion_pixel = torch.nn.L1Loss().to(self.device)
        self.criterion_decision = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(opt.lambda_recall)).to(self.device)
        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

    def param2stroke(self, param, H, W):
        param_list = torch.split(param, [self.d_shape] + [1] * (self.d - self.d_shape), dim=1)
        actions, R, G, B, = param_list
        brush = self.gan_strokes.forward(actions)
        brush = transforms.Resize((H, W))(brush)
        alphas = torch.cat([brush, brush, brush], dim=1)
        # Clear noise from alpha
        quantiles = torch.quantile(alphas.reshape(alphas.shape[0], -1), 0.8, dim=1, keepdim=True)
        alphas = (alphas > quantiles[:, None, None]).float()
        color_map = torch.stack([R, G, B], dim=1)
        color_map = color_map.unsqueeze(-1).repeat(1, 1, 1, brush.shape[3])
        brush = brush * color_map
        return brush, alphas

    def set_input(self):
        with torch.no_grad():
            old_param = torch.rand(self.opt.batch_size // 4, self.opt.used_strokes, self.d, device=self.device)
            old_param = old_param.view(-1, self.d).contiguous()
            foregrounds, alphas = self.param2stroke(old_param, self.patch_size * 2, self.patch_size * 2)
            foregrounds = morphology.Dilation2d(m=1)(foregrounds)
            alphas = morphology.Erosion2d(m=1)(alphas)
            foregrounds = foregrounds.view(self.opt.batch_size // 4, self.opt.used_strokes, 3, self.patch_size * 2,
                                           self.patch_size * 2).contiguous()
            alphas = alphas.view(self.opt.batch_size // 4, self.opt.used_strokes, 3, self.patch_size * 2,
                                 self.patch_size * 2).contiguous()
            old = torch.zeros(self.opt.batch_size // 4, 3, self.patch_size * 2, self.patch_size * 2, device=self.device)
            for i in range(self.opt.used_strokes):
                foreground = foregrounds[:, i, :, :, :]
                alpha = alphas[:, i, :, :, :]
                old = foreground * alpha + old * (1 - alpha)
            old = old.view(self.opt.batch_size // 4, 3, 2, self.patch_size, 2, self.patch_size).contiguous()
            old = old.permute(0, 2, 4, 1, 3, 5).contiguous()
            self.old = old.view(self.opt.batch_size, 3, self.patch_size, self.patch_size).contiguous()
            gt_param = torch.rand(self.opt.batch_size, self.opt.used_strokes, self.d, device=self.device)
            self.gt_param = gt_param[:, :, :self.d_shape]
            gt_param = gt_param.view(-1, self.d).contiguous()
            foregrounds, alphas = self.param2stroke(gt_param, self.patch_size, self.patch_size)
            foregrounds = morphology.Dilation2d(m=1)(foregrounds)
            alphas = morphology.Erosion2d(m=1)(alphas)
            foregrounds = foregrounds.view(self.opt.batch_size, self.opt.used_strokes, 3, self.patch_size,
                                           self.patch_size).contiguous()
            alphas = alphas.view(self.opt.batch_size, self.opt.used_strokes, 3, self.patch_size,
                                 self.patch_size).contiguous()
            self.render = self.old.clone()
            gt_decision = torch.ones(self.opt.batch_size, self.opt.used_strokes, device=self.device)
            for i in range(self.opt.used_strokes):
                foreground = foregrounds[:, i, :, :, :]
                alpha = alphas[:, i, :, :, :]
                for j in range(i):
                    iou = (torch.sum(alpha * alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5) / (
                            torch.sum(alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5)
                    gt_decision[:, i] = ((iou < 0.75) | (~gt_decision[:, j].bool())).float() * gt_decision[:, i]
                decision = gt_decision[:, i].view(self.opt.batch_size, 1, 1, 1).contiguous()
                self.render = foreground * alpha * decision + self.render * (1 - alpha * decision)
            self.gt_decision = gt_decision

    def forward(self):
        param, decisions = self.net_g(self.render, self.old)
        # stroke_param: b, stroke_per_patch, param_per_stroke
        # decision: b, stroke_per_patch, 1
        self.pred_decision = decisions.view(-1, self.opt.used_strokes).contiguous()
        self.pred_param = param[:, :, :self.d_shape]
        param = param.view(-1, self.d).contiguous()
        foregrounds, alphas = self.param2stroke(param, self.patch_size, self.patch_size)
        foregrounds = morphology.Dilation2d(m=1)(foregrounds)
        alphas = morphology.Erosion2d(m=1)(alphas)
        # foreground, alpha: b * stroke_per_patch, 3, output_size, output_size
        foregrounds = foregrounds.view(-1, self.opt.used_strokes, 3, self.patch_size, self.patch_size)
        alphas = alphas.view(-1, self.opt.used_strokes, 3, self.patch_size, self.patch_size)
        # foreground, alpha: b, stroke_per_patch, 3, output_size, output_size
        decisions = networks.SignWithSigmoidGrad.apply(decisions.view(-1, self.opt.used_strokes, 1, 1, 1).contiguous())
        self.rec = self.old.clone()
        for j in range(foregrounds.shape[1]):
            foreground = foregrounds[:, j, :, :, :]
            alpha = alphas[:, j, :, :, :]
            decision = decisions[:, j, :, :, :]
            self.rec = foreground * alpha * decision + self.rec * (1 - alpha * decision)

    @staticmethod
    def get_sigma_sqrt(w, h, theta):
        sigma_00 = w * (torch.cos(theta) ** 2) / 2 + h * (torch.sin(theta) ** 2) / 2
        sigma_01 = (w - h) * torch.cos(theta) * torch.sin(theta) / 2
        sigma_11 = h * (torch.cos(theta) ** 2) / 2 + w * (torch.sin(theta) ** 2) / 2
        sigma_0 = torch.stack([sigma_00, sigma_01], dim=-1)
        sigma_1 = torch.stack([sigma_01, sigma_11], dim=-1)
        sigma = torch.stack([sigma_0, sigma_1], dim=-2)
        return sigma

    @staticmethod
    def get_sigma(w, h, theta):
        sigma_00 = w * w * (torch.cos(theta) ** 2) / 4 + h * h * (torch.sin(theta) ** 2) / 4
        sigma_01 = (w * w - h * h) * torch.cos(theta) * torch.sin(theta) / 4
        sigma_11 = h * h * (torch.cos(theta) ** 2) / 4 + w * w * (torch.sin(theta) ** 2) / 4
        sigma_0 = torch.stack([sigma_00, sigma_01], dim=-1)
        sigma_1 = torch.stack([sigma_01, sigma_11], dim=-1)
        sigma = torch.stack([sigma_0, sigma_1], dim=-2)
        return sigma
    
    @staticmethod
    def rotate(coords, alpha):
        x, y = torch.split(coords, 1, dim=-1)
        rotated_x = x * torch.cos(alpha) - y * torch.sin(alpha)
        rotated_y = x * torch.sin(alpha) + y * torch.cos(alpha)
        return torch.stack([rotated_x, rotated_y], dim=-1).squeeze()

    def get_rotated_bounding_box(self, param):
        start, end, control, pressure, entry_pressure, size = torch.split(
            param, (2, 2, 2, 1, 1, 1), dim=-1
        )
        start = torch.clamp_max(start + entry_pressure * 0.15, 1)
        end = torch.clamp_max(end + pressure * 0.15, 1)
        control = torch.clamp_min(control - size * 0.15, 0)
        translation = torch.clone(start)
        end_x, end_y = torch.split(start, 1, dim=-1)
        alpha = torch.atan2(end_y, end_x)
        start = self.rotate(start - translation, -alpha)
        end = self.rotate(end - translation, -alpha)
        control = self.rotate(control - translation, -alpha)
        t = (start - control) / (-2 * control + start + end)
        solution = (1 - t) ** 2 * start + 2 * t * (1 - t) * control + t ** 2 * end
        max = torch.maximum(solution, torch.maximum(start, end))
        min = torch.minimum(solution, torch.minimum(start, end))
        max_x, max_y = torch.split(max, 1, dim=-1)
        min_x, min_y = torch.split(min, 1, dim=-1)
        bbox_max = self.rotate(max, alpha) + translation
        bbox_min = self.rotate(min, alpha) + translation
        bbox_max_adj = self.rotate(torch.stack([max_x, min_y], dim=-1).squeeze(), alpha) + translation
        bbox_min_adj = self.rotate(torch.stack([min_x, max_y], dim=-1).squeeze(), alpha) + translation
        x_center = (bbox_max[..., 0] + bbox_min[..., 0]) / 2
        y_center = (bbox_min_adj[..., 1] + bbox_min[..., 1]) / 2
        w = ((bbox_min_adj[..., 1] - bbox_max[..., 1]) ** 2 + (bbox_min_adj[..., 0] - bbox_max[..., 0]) ** 2) ** 0.5
        h = ((bbox_max[..., 1] - bbox_max_adj[..., 1]) ** 2 + (bbox_max[..., 0] - bbox_max_adj[..., 0]) ** 2) ** 0.5
        angle = torch.atan2(bbox_max_adj[..., 1] - bbox_max[..., 1], bbox_max_adj[..., 0] - bbox_max[..., 0])
        rotated_bbox = torch.stack(
            [torch.clamp(x_center, 0, 1), torch.clamp(y_center, 0, 1), torch.clamp(w, 0, 1), torch.clamp(h, 0, 1), angle]
        )
        return rotated_bbox.T if rotated_bbox.dim() == 2 else rotated_bbox.permute(1, 2, 0)

    def gaussian_w_distance(self, param_1, param_2):
        param_1 = self.get_rotated_bounding_box(param_1)
        mu_1, w_1, h_1, theta_1 = torch.split(param_1, (2, 1, 1, 1), dim=-1)
        w_1 = w_1.squeeze(-1)
        h_1 = h_1.squeeze(-1)
        theta_1 = theta_1.squeeze(-1)
        trace_1 = (w_1 ** 2 + h_1 ** 2) / 4
        param_2 = self.get_rotated_bounding_box(param_2)
        mu_2, w_2, h_2, theta_2 = torch.split(param_2, (2, 1, 1, 1), dim=-1)
        w_2 = w_2.squeeze(-1)
        h_2 = h_2.squeeze(-1)
        theta_2 = theta_2.squeeze(-1)
        trace_2 = (w_2 ** 2 + h_2 ** 2) / 4
        sigma_1_sqrt = self.get_sigma_sqrt(w_1, h_1, theta_1)
        sigma_2 = self.get_sigma(w_2, h_2, theta_2)
        trace_12 = torch.matmul(torch.matmul(sigma_1_sqrt, sigma_2), sigma_1_sqrt)
        trace_12 = torch.sqrt(trace_12[..., 0, 0] + trace_12[..., 1, 1] + 2 * torch.sqrt(
            trace_12[..., 0, 0] * trace_12[..., 1, 1] - trace_12[..., 0, 1] * trace_12[..., 1, 0]))
        distance = torch.sum((mu_1 - mu_2) ** 2, dim=-1) + trace_1 + trace_2 - 2 * trace_12
        distance = torch.nan_to_num(distance, 1e4)
        return distance

    def optimize_parameters(self):
        self.forward()
        self.loss_pixel = self.criterion_pixel(self.rec, self.render) * self.opt.lambda_pixel
        cur_valid_gt_size = 0
        with torch.no_grad():
            r_idx = []
            c_idx = []
            for i in range(self.gt_param.shape[0]):
                is_valid_gt = self.gt_decision[i].bool()
                valid_gt_param = self.gt_param[i, is_valid_gt]
                cost_matrix_l1 = torch.cdist(self.pred_param[i], valid_gt_param, p=1)
                pred_param_broad = self.pred_param[i].unsqueeze(1).contiguous().repeat(
                    1, valid_gt_param.shape[0], 1)
                valid_gt_param_broad = valid_gt_param.unsqueeze(0).contiguous().repeat(
                    self.pred_param.shape[1], 1, 1)
                cost_matrix_w = self.gaussian_w_distance(pred_param_broad, valid_gt_param_broad)
                decision = self.pred_decision[i]
                cost_matrix_decision = (1 - decision).unsqueeze(-1).repeat(1, valid_gt_param.shape[0])
                try:
                    r, c = linear_sum_assignment((cost_matrix_l1 + cost_matrix_w + cost_matrix_decision).cpu())
                except ValueError:
                    r = np.arange(valid_gt_param.shape[0])
                    c = np.arange(valid_gt_param.shape[0])
                r_idx.append(torch.tensor(r + self.pred_param.shape[1] * i, device=self.device))
                c_idx.append(torch.tensor(c + cur_valid_gt_size, device=self.device))
                cur_valid_gt_size += valid_gt_param.shape[0]
            r_idx = torch.cat(r_idx, dim=0)
            c_idx = torch.cat(c_idx, dim=0)
            paired_gt_decision = torch.zeros(self.gt_decision.shape[0] * self.gt_decision.shape[1], device=self.device)
            paired_gt_decision[r_idx] = 1.
        all_valid_gt_param = self.gt_param[self.gt_decision.bool(), :]
        all_pred_param = self.pred_param.view(-1, self.pred_param.shape[2]).contiguous()
        all_pred_decision = self.pred_decision.view(-1).contiguous()
        paired_gt_param = all_valid_gt_param[c_idx, :]
        paired_pred_param = all_pred_param[r_idx, :]
        self.loss_gt = self.criterion_pixel(paired_pred_param, paired_gt_param) * self.opt.lambda_gt
        self.loss_w = self.gaussian_w_distance(paired_pred_param, paired_gt_param).mean() * self.opt.lambda_w
        self.loss_decision = self.criterion_decision(all_pred_decision, paired_gt_decision) * self.opt.lambda_decision
        loss = self.loss_pixel + self.loss_gt + self.loss_w + self.loss_decision
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
