from typing import List
from manimlib import *
from collections import Counter, defaultdict

# from manim import *

from numba import njit, prange
from random import randint
from numpy.polynomial import Polynomial

import sympy as sp
import numpy as np

BAIL_OUT_R2 = 4


def get_sorted_prime_divisors(n: int):
    divisors = []
    i = 2
    while n > 1:
        if n % i == 0:
            divisors.append(i)
            n = n / i
        else:
            i += 1

    return divisors


def periodic_divisors(n: int):
    return [i for i in range(1, n // 2 + 1) if n % i == 0]


def test_periodic_divisors():
    assert sorted(periodic_divisors(1)) == []
    assert sorted(periodic_divisors(2)) == [1]
    assert sorted(periodic_divisors(3)) == [1]
    assert sorted(periodic_divisors(4)) == [1, 2]
    assert sorted(periodic_divisors(5)) == [1]
    assert sorted(periodic_divisors(6)) == [1, 2, 3]
    assert sorted(periodic_divisors(8)) == [1, 2, 4]
    assert sorted(periodic_divisors(100)) == [1, 2, 4, 5, 10, 20, 25, 50]


def get_hyperbolic_centers(max_period: int) -> List[List[np.complex_]]:
    x = sp.var("x")
    p = x

    reduced_polys = []
    roots = []
    for period in range(1, max_period):
        reduced_poly = p
        for i in periodic_divisors(period):
            q, r = sp.div(reduced_poly, reduced_polys[i - 1])
            assert r == 0
            reduced_poly = q

        # roots.append(np.complex128(sp.solve(reduced_poly, minimal=True)))
        roots.append(np.roots(sp.Poly(reduced_poly).all_coeffs()))
        poly = sp.Poly(reduced_poly)
        reduced_polys.append(reduced_poly)
        p = p * p + x

    return roots


def test_hyperbolic_centers():
    for i, roots in enumerate(get_hyperbolic_centers(10)):
        init = roots.copy()
        for _ in range(i):
            roots = roots * roots + init

        assert np.max(np.abs(roots)) < 0.001


def get_approx_hyperbolic_bulbs(max_period):
    for q in range(2, max_period):
        out = []
        for p in range(1, q):
            if np.gcd(p, q) != 1:
                continue
            t = np.pi * p / q
            z = np.exp(2 * t * 1j) / 2
            z = z * (1 - z)
            tangent_perpendicular = 1 / 2 * (np.exp(2 * t * 1j) - np.exp(4 * t * 1j))
            tangent_perpendicular /= np.abs(tangent_perpendicular)
            r = 1 / (q * q) * np.sin(t)
            z += r * tangent_perpendicular
            circle = Circle(radius=r, stroke_width=np.sqrt(r)).move_to([z.real, z.imag, 0])
            out.append(circle)
        yield out


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
        # self.remove(path)
        self.play(FadeOut(construction))

    def construct(self):
        super().construct()

        # self.outline_main_bulb()

        for circles in get_approx_hyperbolic_bulbs(1000):
            self.play(FadeIn(VGroup(*circles)), run_time=0.25)

        self.wait(10)

    def transform_unit_disk(self):
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
    # test_periodic_divisors()
    # test_hyperbolic_centers()
    Cardioid().construct()