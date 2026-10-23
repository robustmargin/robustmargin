"""``_step_supremum_log_likelihood`` against a brute-force adjacent-only reference.

The reference enumerates, for the levels ordered by severity, every split with no free level and
every free level ``j`` at the transition (``[0, j)`` at ``u``, ``(j, m)`` at ``l``, level ``j`` at
its own rate clipped to ``[l, u]``), with each estimated asymptote its side's pooled rate under
``l <= u``. A log-axis control (severity 0) is never free. A free level that is not at the
transition is not a limit of any monotone curve, so it is not enumerated.

The supremum is returned as a ``(value, degenerate)`` pair, ``degenerate`` being whether the best
configuration has ``u == l`` (the constant curve, the ``s -> infinity`` limit).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.stats import binom

from robustmargin import models as models_module


def _group(k: list[int], n: list[int], p: float) -> float:
    return float(sum(binom.logpmf(ki, ni, p) for ki, ni in zip(k, n, strict=True)))


def reference_supremum(
    severity: list[float],
    successes: list[int],
    trials: list[int],
    *,
    estimate_lower: bool,
    estimate_upper: bool,
    lower_value: float,
    upper_value: float,
    log_axis: bool,
) -> tuple[float, bool | None]:
    """``(supremum, degenerate)``; ``degenerate`` is ``None`` when configurations of both kinds
    attain the supremum, so the flag is not determined."""
    order = sorted(range(len(severity)), key=lambda i: severity[i])
    sev = [severity[i] for i in order]
    k = [successes[i] for i in order]
    n = [trials[i] for i in order]
    m = len(k)
    control = log_axis and sev[0] == 0.0

    def pooled(idx: list[int]) -> float:
        return sum(k[i] for i in idx) / sum(n[i] for i in idx)

    configs: list[tuple[list[int], int | None, list[int]]] = [
        (list(range(s)), None, list(range(s, m))) for s in range(1, m)
    ]
    for j in range(m):
        left, right = list(range(j)), list(range(j + 1, m))
        if (control and j == 0) or (not left and estimate_upper) or (not right and estimate_lower):
            continue
        configs.append((left, j, right))

    results: list[tuple[float, bool]] = []
    for left, free, right in configs:
        u = pooled(left) if estimate_upper else upper_value
        low = pooled(right) if estimate_lower else lower_value
        if estimate_lower and estimate_upper:
            if low > u:
                u = low = pooled(list(range(m)))
        elif estimate_lower:
            low = min(low, u)
        elif estimate_upper:
            u = max(u, low)
        total = _group([k[i] for i in left], [n[i] for i in left], u)
        total += _group([k[i] for i in right], [n[i] for i in right], low)
        if free is not None:
            total += _group([k[free]], [n[free]], min(max(k[free] / n[free], low), u))
        results.append((total, abs(u - low) <= 1e-12))
    best = max(total for total, _ in results)
    kinds = {flag for total, flag in results if total >= best - 1e-9}
    return best, (kinds.pop() if len(kinds) == 1 else None)


def _supremum(
    severity: list[float],
    successes: list[int],
    trials: list[int],
    *,
    estimate_lower: bool,
    estimate_upper: bool,
    lower_value: float,
    upper_value: float,
    log_axis: bool,
) -> tuple[float, bool]:
    result = models_module._step_supremum_log_likelihood(
        np.asarray(severity, dtype=np.float64),
        np.asarray(successes, dtype=np.float64),
        np.asarray(trials, dtype=np.float64),
        estimate_lower=estimate_lower,
        estimate_upper=estimate_upper,
        lower_value=lower_value,
        upper_value=upper_value,
        scale="log" if log_axis else "linear",
    )
    value, degenerate = result
    return float(value), bool(degenerate)


_CONFIGURATIONS = {
    "both_estimated": (True, True, 0.0, 1.0),
    "upper_fixed_lower_estimated": (True, False, 0.0, 0.9),
    "upper_estimated_lower_fixed": (False, True, 0.15, 1.0),
    "both_fixed": (False, False, 0.15, 0.9),
    "log_axis_with_control": (True, True, 0.0, 1.0),
}


@st.composite
def _designs(draw: st.DrawFn) -> tuple[str, list[float], list[int], list[int]]:
    name = draw(st.sampled_from(sorted(_CONFIGURATIONS)))
    count = draw(st.integers(min_value=2, max_value=6))
    trials = draw(st.lists(st.integers(1, 30), min_size=count, max_size=count))
    successes = [draw(st.integers(0, n)) for n in trials]
    severity = [float(i + 1) for i in range(count)]
    if name == "log_axis_with_control":
        severity[0] = 0.0
    return name, severity, successes, trials


@settings(
    max_examples=150,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(_designs())
def test_supremum_equals_the_brute_force_adjacent_only_reference(
    design: tuple[str, list[float], list[int], list[int]],
) -> None:
    name, severity, successes, trials = design
    estimate_lower, estimate_upper, lower_value, upper_value = _CONFIGURATIONS[name]
    kwargs = {
        "estimate_lower": estimate_lower,
        "estimate_upper": estimate_upper,
        "lower_value": lower_value,
        "upper_value": upper_value,
        "log_axis": name == "log_axis_with_control",
    }

    actual, degenerate = _supremum(severity, successes, trials, **kwargs)
    expected, expected_degenerate = reference_supremum(severity, successes, trials, **kwargs)

    if math.isinf(expected):
        assert actual == expected
    else:
        assert actual == pytest.approx(expected, abs=1e-9)
        if expected_degenerate is not None:
            assert degenerate is expected_degenerate


def test_supremum_does_not_use_a_free_level_away_from_the_transition() -> None:
    # A middle level at 800/1000 between two levels near 0.95 and then a decline: freeing it is
    # not reachable by a monotone curve, so the supremum is the adjacent-only value.
    severity = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    successes = [19, 800, 19, 14, 6, 1]
    trials = [20, 1000, 20, 20, 20, 20]

    value, _ = _supremum(
        severity,
        successes,
        trials,
        estimate_lower=True,
        estimate_upper=False,
        lower_value=0.0,
        upper_value=0.95,
        log_axis=False,
    )

    assert value == pytest.approx(-32.76, abs=0.01)
    assert value == pytest.approx(
        reference_supremum(
            severity,
            successes,
            trials,
            estimate_lower=True,
            estimate_upper=False,
            lower_value=0.0,
            upper_value=0.95,
            log_axis=False,
        )[0],
        abs=1e-9,
    )


def test_supremum_never_frees_a_log_axis_control() -> None:
    # A control at 5/20 followed by two low levels: freeing the control would let it sit at its
    # own rate 0.25 with the rest at l = 0.1. The control always takes u = 0.9 instead.
    value, _ = _supremum(
        [0.0, 1.0, 2.0],
        [5, 2, 1],
        [20, 20, 20],
        estimate_lower=False,
        estimate_upper=False,
        lower_value=0.1,
        upper_value=0.9,
        log_axis=True,
    )

    expected = _group([5], [20], 0.9) + _group([2, 1], [20, 20], 0.1)
    assert value == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    ("successes", "kwargs", "degenerate"),
    [
        pytest.param([9, 9, 3, 3], {}, False, id="a_step_down"),
        pytest.param([2, 2, 8, 8], {}, True, id="rising_pools_both_asymptotes"),
        pytest.param([6, 6, 6, 6], {}, True, id="flat_is_the_constant_curve"),
        pytest.param(
            [2, 2, 9, 9],
            {"estimate_upper": False, "upper_value": 0.5},
            True,
            id="estimated_lower_clipped_to_the_fixed_upper",
        ),
        pytest.param(
            [9, 9, 3, 3],
            {
                "estimate_lower": False,
                "estimate_upper": False,
                "lower_value": 0.2,
                "upper_value": 0.8,
            },
            False,
            id="fixed_distinct_asymptotes",
        ),
    ],
)
def test_supremum_reports_whether_its_best_configuration_is_the_constant_curve(
    successes: list[int], kwargs: dict[str, Any], degenerate: bool
) -> None:
    options: dict[str, Any] = {
        "estimate_lower": True,
        "estimate_upper": True,
        "lower_value": 0.0,
        "upper_value": 1.0,
        "log_axis": False,
    }
    options.update(kwargs)

    _, actual = _supremum([1.0, 2.0, 3.0, 4.0], successes, [10] * 4, **options)

    assert actual is degenerate


def test_supremum_result_is_a_value_degenerate_pair() -> None:
    result = models_module._step_supremum_log_likelihood(
        np.array([1.0, 2.0]),
        np.array([9.0, 1.0]),
        np.array([10.0, 10.0]),
        estimate_lower=True,
        estimate_upper=True,
        lower_value=0.0,
        upper_value=1.0,
        scale="linear",
    )

    assert result._fields == ("value", "degenerate")
