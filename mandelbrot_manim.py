from manimlib import *

# from manim import *

from numba import njit, prange
from random import randint

import numpy as np

BAIL_OUT_R2 = 4


@njit(parallel=True, fastmath=True)
def mandelbrot(points: np.ndarray):
    max_iter = len(points[0][0])

    for i in prange(len(points)):
        for j in range(len(points[i])):
            point = points[i, j]
            c = point[0]
            z = c
            k = 1
            while (k < max_iter) and (z.real * z.real + z.imag + z.imag < BAIL_OUT_R2):
                z = z * z + c
                point[k] = z
                k += 1


class MandelbrotGrid(Scene):
    FIXED_RADIUS = 1 / 4
    ROTATING_RADIUS = FIXED_RADIUS

    BROT_FRAME_HEIGHT = 2.5

    def __init__(self, **kwargs):
        super().__init__(
            camera_config={
                "frame_config": {
                    "frame_shape": (
                        self.BROT_FRAME_HEIGHT * ASPECT_RATIO,
                        self.BROT_FRAME_HEIGHT,
                    )
                }
            },
            **kwargs
        )

    def is_point_in_main_cardioid(self, point: np.complex_):
        return np.abs(1 - np.sqrt(1 - 4 * point)) <= 1

    def is_point_in_left_circle(self, point: np.complex_):
        return np.abs(point + 1) <= 1 / 4

    def construct(self):
        super().construct()
        self.plane = ComplexPlane()
        self.play(ShowCreation(self.plane, run_time=1, lag_ratio=0.1))


class MandelbrotManim(MandelbrotGrid):
    def construct(self):
        super().construct()

        x = np.linspace(-1.5, 1, 50)
        y = np.linspace(-1.25, 1.25, 50)
        xx, yy = np.meshgrid(x, y)
        init_points = xx + 1j * yy

        max_iter = 925
        points = np.zeros((*init_points.shape, max_iter), dtype=np.complex_)
        points[:, :, 0] = init_points
        mandelbrot(points)

        shape = points.shape
        points_to_plot = [
            [point, None]
            for row in points
            for point in row
            if not (
                self.is_point_in_main_cardioid(point[0])
                or self.is_point_in_left_circle(point[0])
            )
        ]
        point_radius = 10 / (shape[0] * shape[1])

        for i in range(900, max_iter):
            anims = []
            for data in points_to_plot:
                point_history = data[0]
                dot = data[1]
                point = point_history[i]
                if not point:
                    if dot is not None:
                        self.remove(dot)
                    continue

                coords = self.plane.number_to_point(point)
                if dot is None:
                    dot = Dot(coords, color=YELLOW, radius=point_radius)
                    data[1] = dot
                    self.add(dot)
                else:
                    anims.append(dot.animate.move_to(coords))

            if anims:
                self.play(*anims, run_time=0.5)


class Cardioid(MandelbrotGrid):
    STROKE_WIDTH = 1

    def outline_main_bulb(self):
        path = VMobject()
        dot = SmallDot()
        path.set_points_as_corners([dot.get_center(), dot.get_center()])
        path.add_updater(lambda path: path.add_points_as_corners([dot.get_center()]))
        self.add(path)

        fixed_circle = Circle(
            radius=self.FIXED_RADIUS, stroke_width=self.STROKE_WIDTH
        ).move_to(ORIGIN)
        rotating_circle = Circle(
            radius=self.ROTATING_RADIUS, stroke_width=self.STROKE_WIDTH
        ).move_to(RIGHT * (self.FIXED_RADIUS + self.ROTATING_RADIUS))

        construction = VGroup(fixed_circle, rotating_circle, dot)

        RADIANS_PER_SEC = TAU / 4
        TOTAL_ROTATION = TAU

        dot.angle = PI

        def rotate_dot(dot: Dot, dt: float):
            dot.angle = (dot.angle + RADIANS_PER_SEC * dt) % TAU
            dot.move_to(rotating_circle.point_from_proportion(dot.angle / TAU))

        dot.add_updater(rotate_dot)

        self.play(FadeIn(construction))
        self.play(
            Rotating(
                rotating_circle,
                angle=TOTAL_ROTATION,
                about_point=fixed_circle.get_center(),
                run_time=TOTAL_ROTATION / RADIANS_PER_SEC,
            )
        )
        dot.remove_updater(rotate_dot)
        self.remove(path)
        self.play(FadeOut(construction))

    def construct(self):
        super().construct()

        x = np.linspace(-1, 1, 25)
        y = np.linspace(-1, 1, 25)
        xx, yy = np.meshgrid(x, y)
        points = xx + 1j * yy
        unit_disk_dots = VGroup(
            *(
                Dot(self.plane.number_to_point(point), radius=0.001)
                for row in points
                for point in row
                if np.abs(point) < 1
            )
        )

        self.outline_main_bulb()

        unit_circle_desc = (
            VGroup(
                Tex(
                    r"""\text{These are some points on the unit disk} \hspace{1cm} \{ z \mid \abs{z} \leq 1 \}"""
                ),
            )
            .arrange(DOWN, buff=1)
            .to_corner(UP + LEFT)
            .scale(0.5)
        )
        self.play(ShowCreation(unit_circle_desc))

        unit_circle = Circle(radius=1, stroke_width=self.STROKE_WIDTH).move_to(ORIGIN)
        transform_group = VGroup(unit_disk_dots, unit_circle)
        self.play(FadeIn(transform_group))

        transform = lambda z: z / 2 * (1 - z / 2)
        self.play(
            transform_group.animate.apply_complex_function(transform),
            run_time=5,
        )

        self.plane.prepare_for_nonlinear_transform()
        self.play(self.plane.animate.apply_complex_function(transform), run_time=5)


if __name__ == "__main__":
    Cardioid().construct()