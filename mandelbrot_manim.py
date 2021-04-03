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


if __name__ == "__main__":
    MandelbrotManim().construct()