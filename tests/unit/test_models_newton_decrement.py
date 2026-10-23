"""``_newton_decrement_half``: the scale-free optimality measure ``lambda^2 / 2`` with
``lambda^2 = g' (-H)^-1 g`` from numerical derivatives.

Its finite-difference step must follow the curvature of the function (a fraction of each
coordinate's standard error), not the size of the coordinate: a step of ``1e-4 * |x|`` is a large
fraction of the standard error on a steep curve, where truncation error then makes the decrement
exceed the acceptance tolerance at a true maximum.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from robustmargin import models as models_module

_TOLERANCE = 1e-6  # the acceptance tolerance of the fit paths


def _poisson_like(
    counts: NDArray[np.float64], theta: NDArray[np.float64]
) -> Callable[[NDArray[np.float64]], float]:
    """``sum_i N_i (log(x_i / theta_i) - x_i / theta_i + 1)``: maximum 0 at ``x = theta``,
    gradient ``N_i (1/x_i - 1/theta_i)``, Hessian ``-N_i / x_i^2``, and a standard error of
    ``theta_i / sqrt(N_i)``: steep and asymmetric when ``theta`` is small and ``N`` large.
    """

    def func(x: NDArray[np.float64]) -> float:
        return float(np.sum(counts * (np.log(x / theta) - x / theta + 1.0)))

    return func


def test_decrement_of_a_quadratic_equals_the_analytic_value() -> None:
    root = np.array([[2.0, 0.0, 0.0], [0.7, 1.5, 0.0], [-0.4, 0.3, 3.0]])
    information = root @ root.T
    optimum = np.array([1000.0, -250.0, 3.0])
    point = optimum + np.array([0.3, -0.2, 0.1])

    def func(x: NDArray[np.float64]) -> float:
        delta = x - optimum
        return float(-0.5 * delta @ information @ delta)

    decrement = models_module._newton_decrement_half(func, point)

    delta = point - optimum
    assert decrement == pytest.approx(0.5 * delta @ information @ delta, rel=1e-5)


def test_decrement_of_a_quadratic_at_its_maximum_is_negligible() -> None:
    information = np.array([[4.0, 1.0], [1.0, 2.0]])
    optimum = np.array([5.0, -2000.0])

    def func(x: NDArray[np.float64]) -> float:
        delta = x - optimum
        return float(-0.5 * delta @ information @ delta)

    assert models_module._newton_decrement_half(func, optimum) < 1e-9


def test_decrement_is_infinite_when_the_information_is_not_positive_definite() -> None:
    def saddle(x: NDArray[np.float64]) -> float:
        return float(x[0] ** 2 - x[1] ** 2)

    assert models_module._newton_decrement_half(saddle, np.array([0.1, 0.1])) == float("inf")


def test_decrement_at_the_maximum_of_a_steep_badly_scaled_function_is_below_1e_9() -> None:
    theta = np.array([0.01])
    func = _poisson_like(np.array([1e5]), theta)

    decrement = models_module._newton_decrement_half(func, theta.copy())

    assert decrement < 1e-9


def test_decrement_at_the_maximum_with_coordinates_on_very_different_scales_is_below_1e_9() -> None:
    theta = np.array([0.01, 500.0, 2.0])
    func = _poisson_like(np.array([1e5, 1e3, 1e4]), theta)

    decrement = models_module._newton_decrement_half(func, theta.copy())

    assert decrement < 1e-9


def test_decrement_off_the_maximum_of_a_steep_function_equals_the_analytic_value() -> None:
    theta = np.array([0.01])
    counts = np.array([1e5])
    point = 1.001 * theta

    decrement = models_module._newton_decrement_half(_poisson_like(counts, theta), point)

    # g = N (1/x - 1/theta), -H = N / x^2, so lambda^2 / 2 = N (1 - x/theta)^2 / 2.
    expected = 0.5 * counts[0] * (1.0 - point[0] / theta[0]) ** 2
    assert decrement == pytest.approx(expected, rel=1e-3)


def test_steep_maximum_is_accepted_by_the_fit_paths_tolerance() -> None:
    theta = np.array([0.01])

    decrement = models_module._newton_decrement_half(
        _poisson_like(np.array([1e5]), theta), theta.copy()
    )

    assert decrement <= _TOLERANCE


def test_decrement_at_a_maximum_with_a_weakly_identified_coordinate_is_accepted() -> None:
    # Coordinate 0 has curvature 1e-8 (standard error 1e4) and a cubic term, so a central
    # difference with a step of 1e-2 standard errors (100) reads a gradient of 1e-5 at the true
    # maximum x = 0, while the default step 1e-4 reads 1e-17. The second pass may only shrink
    # the step, so the default step stands and the maximum is accepted.
    curvature, cubic = 1e-8, 1e-9

    def func(x: NDArray[np.float64]) -> float:
        return float(-0.5 * curvature * x[0] ** 2 + cubic * x[0] ** 3 - 50.0 * (x[1] - 3.0) ** 2)

    decrement = models_module._newton_decrement_half(func, np.array([0.0, 3.0]))

    assert decrement <= _TOLERANCE
