import time
import os
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

from torch.utils.tensorboard import SummaryWriter

# losses: same format as |losses| of plot_current_losses
def print_current_losses(log_name, epoch, iters, losses, t_comp):
    """print current losses on console; also save the losses to the disk

    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time per data point (normalized by batch_size)
    """
    message = "(epoch: %d, iters: %d, time: %.3f) " % (
        epoch,
        iters,
        t_comp,
    )
    for k, v in losses.items():
        message += "%s: %.3f " % (k, v)

    print(message)  # print the message
    with open(log_name, "a") as log_file:
        log_file.write("%s\n" % message)  # save the message


if __name__ == "__main__":
    opt = TrainOptions().parse()  # get training options
    writer = SummaryWriter(
        os.path.join(opt.checkpoints_dir, opt.name, "tensorboard_log")
    )
    log_name = os.path.join(opt.checkpoints_dir, opt.name, "loss_log.txt")
    dataset = create_dataset(
        opt
    )  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print("The number of training images = %d" % dataset_size)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    total_iters = 0  # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        for i in range(dataset_size):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input()  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if (
                total_iters % opt.display_freq == 0
            ):  # display images on tensorboard
                model.compute_visuals()
                visuals = model.get_current_visuals()
                for label, image in visuals.items():
                    writer.add_images(label, image[0], total_iters, dataformats="CHW")

            if (
                total_iters % opt.print_freq == 0
            ):  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                print_current_losses(
                    log_name, epoch, epoch_iter, losses, t_comp
                )
                for label, loss in losses.items():
                    writer.add_scalar(label, loss, total_iters)

            if (
                total_iters % opt.save_latest_freq == 0
            ):  # cache our latest model every <save_latest_freq> iterations
                print(
                    "saving the latest model (epoch %d, total_iters %d)"
                    % (epoch, total_iters)
                )
                save_suffix = "iter_%d" % total_iters if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if (
            epoch % opt.save_epoch_freq == 0
        ):  # cache our model every <save_epoch_freq> epochs
            print(
                "saving the model at the end of epoch %d, iters %d"
                % (epoch, total_iters)
            )
            model.save_networks("latest")
            model.save_networks(epoch)

        print(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time)
        )
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.
