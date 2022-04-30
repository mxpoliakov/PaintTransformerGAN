# Based on https://colab.research.google.com/github/reiinakano/neural-painters/blob/master/notebooks/generate_stroke_examples.ipynb

from lib import surface, tiledsurface, brush

import torch
import numpy as np
from PIL import Image


def point_on_curve_1(t, cx, cy, sx, sy, x1, y1, x2, y2):
    ratio = t / 100.0
    x3, y3 = multiply_add(sx, sy, x1, y1, ratio)
    x4, y4 = multiply_add(cx, cy, x2, y2, ratio)
    x5, y5 = difference(x3, y3, x4, y4)
    x, y = multiply_add(x3, y3, x5, y5, ratio)
    return x, y


def length_and_normal(x1, y1, x2, y2):
    x, y = difference(x1, y1, x2, y2)
    length = np.sqrt(x * x + y * y)
    if length == 0.0:
        x, y = 0.0, 0.0
    else:
        x, y = x / length, y / length
    return length, x, y


def multiply_add(x1, y1, x2, y2, d):
    x3, y3 = multiply(x2, y2, d)
    x, y = add(x1, y1, x3, y3)
    return x, y


def multiply(x, y, d):
    # Multiply vector
    x = x * d
    y = y * d
    return x, y


def add(x1, y1, x2, y2):
    # Add vectors
    x = x1 + x2
    y = y1 + y2
    return x, y


def difference(x1, y1, x2, y2):
    # Difference in x and y between two points
    x = x2 - x1
    y = y2 - y1
    return x, y


def midpoint(x1, y1, x2, y2):
    # Midpoint between 2 points
    x = (x1 + x2) / 2.0
    y = (y1 + y2) / 2.0
    return x, y


class MyPaintImagesDataLoader:
    def __init__(self, H=32, W=32):
        self.rng = np.random.default_rng(42)
        self.head = 0.25
        self.tail = 0.75
        self.surface = tiledsurface.Surface()
        with open("gan_stroke_generator/brushes/classic/dry_brush.myb") as brush_file:
            self.brush_info = brush.BrushInfo(brush_file.read())
        self.brush = brush.Brush(self.brush_info)
        self.H = H
        self.W = W
        self.num_action = 9
        self.num_images = int(10e9)

    def _stroke_to(self, x, y, pressure):
        duration = 0.1
        self.brush.stroke_to(
            self.surface.backend, x, y, pressure, 0.0, 0.0, duration, 0.0, 0.0, 0.0
        )
        self.surface.end_atomic()
        self.surface.begin_atomic()

    def _line_settings(self, entry_pressure, pressure):
        p2 = (entry_pressure + pressure) / 2
        prange1 = p2 - entry_pressure
        prange2 = pressure - p2
        return p2, prange1, prange2

    def curve(
        self, control_x, control_y, start_x, start_y, ex, ey, entry_pressure, pressure
    ):
        (
            midpoint_p,
            prange1,
            prange2,
        ) = self._line_settings(entry_pressure, pressure)
        points_in_curve = 100
        mx, my = midpoint(start_x, start_y, ex, ey)
        length, nx, ny = length_and_normal(mx, my, control_x, control_y)
        cx, cy = multiply_add(mx, my, nx, ny, length * 2)
        x1, y1 = difference(start_x, start_y, cx, cy)
        x2, y2 = difference(cx, cy, ex, ey)
        head = points_in_curve * self.head
        head_range = int(head) + 1
        tail = points_in_curve * self.tail
        tail_range = int(tail) + 1
        tail_length = points_in_curve - tail

        # Beginning
        px, py = point_on_curve_1(1, cx, cy, start_x, start_y, x1, y1, x2, y2)
        length, nx, ny = length_and_normal(start_x, start_y, px, py)
        bx, by = multiply_add(start_x, start_y, nx, ny, 0.25)
        self._stroke_to(bx, by, entry_pressure)
        pressure = abs(1 / head * prange1 + entry_pressure)
        self._stroke_to(px, py, pressure)

        for i in range(2, head_range):
            px, py = point_on_curve_1(i, cx, cy, start_x, start_y, x1, y1, x2, y2)
            pressure = abs(i / head * prange1 + entry_pressure)
            self._stroke_to(px, py, pressure)

        # Middle
        for i in range(head_range, tail_range):
            px, py = point_on_curve_1(i, cx, cy, start_x, start_y, x1, y1, x2, y2)
            self._stroke_to(px, py, midpoint_p)

        # End
        for i in range(tail_range, points_in_curve + 1):
            px, py = point_on_curve_1(i, cx, cy, start_x, start_y, x1, y1, x2, y2)
            pressure = abs((i - tail) / tail_length * prange2 + midpoint_p)
            self._stroke_to(px, py, pressure)

        return pressure

    def draw_stroke(
        self,
        start_x,
        start_y,
        end_x,
        end_y,
        control_x,
        control_y,
        entry_pressure,
        pressure,
        size,
        color_rgb,
    ):
        start_x = start_x * self.H
        start_y = start_y * self.W
        end_x = end_x * self.H
        end_y = end_y * self.W
        control_x = control_x * self.H
        control_y = control_y * self.W

        self.brush.brushinfo.set_color_rgb(color_rgb)
        self.brush.brushinfo.set_base_value("radius_logarithmic", size)
        # Move brush to starting point without leaving it on the canvas.
        self._stroke_to(start_x, start_y, 0)
        self.curve(
            control_x,
            control_y,
            start_x,
            start_y,
            end_x,
            end_y,
            entry_pressure,
            pressure,
        )
        # Relieve brush pressure for next jump
        self._stroke_to(end_x, end_y, 0)
        self.surface.end_atomic()
        self.surface.begin_atomic()

    def get_mypaint_image(
        self,
        start_x,
        start_y,
        end_x,
        end_y,
        control_x,
        control_y,
        entry_pressure,
        pressure,
        size,
        color_rgb,
    ):
        self.draw_stroke(
            start_x,
            start_y,
            end_x,
            end_y,
            control_x,
            control_y,
            entry_pressure,
            pressure,
            size,
            color_rgb,
        )

        rect = [0, 0, self.H, self.W]
        scanline_strips = surface.scanline_strips_iter(self.surface, rect, single_tile_pattern=True)
        img = next(scanline_strips)
        self.surface.clear()
        self.surface.end_atomic()
        self.surface.begin_atomic()

        return img

    def random_action(self):
        return self.rng.uniform(size=[self.num_action])

    def __len__(self):
        return self.num_images

    def __iter__(self):
        for _ in range(self.num_images):
            action = self.random_action()
            img = self.get_mypaint_image(
                start_x=action[0],
                start_y=action[1],
                end_x=action[2],
                end_y=action[3],
                control_x=action[4],
                control_y=action[5],
                pressure=action[6],
                entry_pressure=action[7],
                size=action[8],
                color_rgb=[1, 1, 1],
            )
            img = Image.fromarray(img).convert('L')
            # We need to create batch of size 1
            img = np.expand_dims(img, axis=0)
            # We need to create a channel for img
            img = np.expand_dims(img, axis=0)
            action = np.expand_dims(action, axis=0)
            yield {
                "stroke": torch.from_numpy(img.astype(float) / 255.0),
                "action": torch.from_numpy(action),
            }
