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
    def construct(self):
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
            [points[randint(0, shape[0] - 1), randint(0, shape[1] - 1)], None]
            for _ in range(300)
        ]
        points_to_plot = [[point, None] for row in points for point in row]
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
    def outline_main_bulb(self):
        path = VMobject()
        dot = Dot()
        path.set_points_as_corners([dot.get_center(), dot.get_center()])
        path.add_updater(lambda path: path.add_points_as_corners([dot.get_center()]))
        self.add(path, dot)

        FIXED_RADIUS = 1 / 4
        ROTATING_RADIUS = FIXED_RADIUS
        fixed_circle = Circle(radius=FIXED_RADIUS).move_to(ORIGIN)
        rotating_circle = Circle(radius=ROTATING_RADIUS).move_to(
            RIGHT * (FIXED_RADIUS + ROTATING_RADIUS)
        )
        self.add(fixed_circle, rotating_circle)

        RADIANS_PER_SEC = TAU / 4
        TOTAL_ROTATION = TAU

        dot.angle = PI

        def rotate_dot(dot: Dot, dt: float):
            dot.angle = (dot.angle + RADIANS_PER_SEC * dt) % TAU
            dot.move_to(rotating_circle.point_from_proportion(dot.angle / TAU))

        dot.add_updater(rotate_dot)

        self.play(
            Rotating(
                rotating_circle,
                angle=TOTAL_ROTATION,
                about_point=fixed_circle.get_center(),
                run_time=TOTAL_ROTATION / RADIANS_PER_SEC,
            )
        )
        dot.remove_updater(rotate_dot)

    def construct(self):
        super().construct()

        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        xx, yy = np.meshgrid(x, y)
        points = xx + 1j * yy
        points_and_dots = [
            [point, Dot(self.plane.number_to_point(point), radius=0.001)]
            for row in points
            for point in row
            if np.abs(point) < 1
        ]

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
        self.add(*(dot for _, dot in points_and_dots))

        self.play(
            *(
                dot.animate.move_to(
                    self.plane.number_to_point(point / 2 * (1 - point / 2))
                )
                for point, dot in points_and_dots
            )
        )


if __name__ == "__main__":
    Cardioid().construct()