"""What a bootstrap does with the replicates that cannot be solved.

* **A failed refit** (``SEPARATION``, ``NOT_CONVERGED``, ``CONTROL_INCOMPATIBLE``) is dropped and
  counted in ``BootstrapSummary.failed``. Above 1% failed the result carries a warning starting
  ``BOOTSTRAP_FAILURES:`` that states ``failed/n_resamples``. Above 10% failed the bootstrap gives
  up: the threshold is ``NOT_CONVERGED`` with ``interval_method="exact_bound"``, no point estimate,
  and the summary is still attached.
* **An unreachable replicate** (a target the replicate's curve never crosses, or a value at or
  below zero on a linear axis) is not dropped: it counts as ``-inf`` or ``+inf`` by the side it
  implies and takes part in the percentiles. A side whose unreachable count exceeds
  ``alpha / 2`` of the valid replicates is open. One side open gives ``OPEN_UPPER`` or
  ``OPEN_LOWER`` with the other bound closed; both open gives ``Censoring.NONE`` with no bounds and
  a warning, and the point estimate is kept.

The data here genuinely produce each situation, so no internal function is replaced. The replicate
fractions the thresholds rely on were measured by simulating the same refits: 3.7% of replicate
refits separate on the sparse series (so the 1% and 10% cut-offs are both many standard errors
away at 400 replicates), 43% separate on the steeper one, about 10% of linear-axis replicates put
the midpoint at or below zero, and the isotonic series put a quarter to a third of their replicates
on each unreachable side.
"""

from __future__ import annotations

import pytest

from robustmargin import (
    Axis,
    Censoring,
    Definition,
    Fit,
    Observations,
    Status,
    Threshold,
    fit_dose_response,
    threshold,
)

_LOG = Axis(name="severity", unit="unit", scale="log")
_LINEAR = Axis(name="severity", unit="unit", scale="linear")
_LEVEL = 0.95


def _fit(
    severity: list[float],
    successes: list[int],
    trials: list[int],
    *,
    axis: Axis = _LOG,
    model: str = "binomial",
) -> Fit:
    obs = Observations.from_counts(
        axis,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )
    if model == "isotonic":
        fit = fit_dose_response(obs, model="isotonic")
    else:
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
    assert fit.status is Status.OK
    return fit


def _boot(fit: Fit, method: str, n_resamples: int, seed: int, target: float = 0.5) -> Threshold:
    return threshold(
        fit,
        definition=Definition.absolute(target),
        interval_method=method,
        level=_LEVEL,
        n_resamples=n_resamples,
        seed=seed,
    )


_SPARSE_SEVERITY = [0.0, 0.5, 1.0, 2.0, 4.0]


class TestFailedReplicates:
    def test_more_than_one_percent_failed_adds_a_warning_that_states_the_fraction(self) -> None:
        fit = _fit(_SPARSE_SEVERITY, [6, 5, 3, 2, 0], [6] * 5)

        t = _boot(fit, "parametric_bootstrap", 400, seed=2026)

        assert t.bootstrap is not None
        assert 0.01 * 400 < t.bootstrap.failed <= 0.10 * 400
        assert t.status is Status.OK and t.value is not None
        assert t.interval_method == "parametric_bootstrap"
        messages = [w for w in t.warnings if w.startswith("BOOTSTRAP_FAILURES:")]
        assert len(messages) == 1
        assert f"{t.bootstrap.failed}/400" in messages[0]

    def test_up_to_ten_percent_failed_still_returns_an_interval_from_the_valid_replicates(
        self,
    ) -> None:
        fit = _fit(_SPARSE_SEVERITY, [6, 5, 3, 2, 0], [6] * 5)

        t = _boot(fit, "parametric_bootstrap", 400, seed=2026)

        assert t.lo is not None and t.value is not None and t.hi is not None
        assert t.lo <= t.value <= t.hi

    def test_more_than_ten_percent_failed_gives_up_with_no_number(self) -> None:
        fit = _fit(_SPARSE_SEVERITY, [6, 6, 4, 1, 0], [6] * 5)

        t = _boot(fit, "parametric_bootstrap", 300, seed=2026)

        assert t.status is Status.NOT_CONVERGED
        assert t.interval_method == "exact_bound"
        assert t.value is None
        assert t.censoring in (Censoring.NONE, Censoring.RIGHT, Censoring.LEFT)
        assert t.bootstrap is not None
        assert t.bootstrap.n_resamples == 300 and t.bootstrap.failed > 0.10 * 300

    def test_a_give_up_still_reports_how_many_replicates_failed(self) -> None:
        fit = _fit(_SPARSE_SEVERITY, [6, 6, 4, 1, 0], [6] * 5)

        t = _boot(fit, "parametric_bootstrap", 300, seed=2026)

        assert t.bootstrap is not None
        assert (
            t.bootstrap.failed + t.bootstrap.unreachable_low + t.bootstrap.unreachable_high <= 300
        )
        assert t.bootstrap.seed == 2026

    def test_a_give_up_is_reproducible_under_its_seed(self) -> None:
        fit = _fit(_SPARSE_SEVERITY, [6, 6, 4, 1, 0], [6] * 5)

        assert _boot(fit, "parametric_bootstrap", 300, seed=2026) == _boot(
            fit, "parametric_bootstrap", 300, seed=2026
        )

    def test_a_clean_run_has_no_failure_warning(self) -> None:
        fit = _fit(
            [0.0, 0.2, 0.4, 0.8, 1.6, 3.2],
            [400, 392, 340, 200, 60, 8],
            [400] * 6,
        )

        t = _boot(fit, "parametric_bootstrap", 99, seed=1)

        assert t.bootstrap is not None and t.bootstrap.failed == 0
        assert not any(w.startswith("BOOTSTRAP_FAILURES:") for w in t.warnings)


class TestUnreachableReplicatesCountBySide:
    def test_a_linear_axis_midpoint_at_or_below_zero_counts_low_and_opens_the_lower_side(
        self,
    ) -> None:
        fit = _fit(
            [0.0, 0.5, 1.0, 2.0, 3.0, 4.0],
            [54, 35, 18, 3, 0, 0],
            [100] * 6,
            axis=_LINEAR,
        )

        t = _boot(fit, "parametric_bootstrap", 300, seed=1)

        assert t.status is Status.OK and t.censoring is Censoring.OPEN_LOWER
        assert t.lo is None and t.hi is not None and t.value is not None
        assert 0.0 < t.value <= t.hi
        assert t.interval_method == "parametric_bootstrap"
        assert t.bootstrap is not None
        assert t.bootstrap.unreachable_low > 0.025 * 300
        assert t.bootstrap.unreachable_high == 0

    def test_both_sides_open_gives_no_censoring_label_no_bounds_and_a_warning(self) -> None:
        fit = _fit(
            [0.0, 0.1, 0.5, 2.5],
            [20, 11, 9, 9],
            [20] * 4,
            model="isotonic",
        )

        t = _boot(fit, "cell_bootstrap", 999, seed=1)

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.value is not None and t.lo is None and t.hi is None
        assert t.interval_method == "cell_bootstrap"
        assert t.warnings
        assert t.bootstrap is not None
        assert t.bootstrap.unreachable_low > 0.025 * 999
        assert t.bootstrap.unreachable_high > 0.025 * 999

    def test_a_side_with_no_unreachable_replicates_stays_closed(self) -> None:
        fit = _fit(
            [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
            [200, 196, 190, 170, 120, 70, 20],
            [200] * 7,
            model="isotonic",
        )

        t = _boot(fit, "cell_bootstrap", 499, seed=1)

        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.lo is not None and t.hi is not None
        assert t.bootstrap is not None
        assert t.bootstrap.unreachable_low == 0 and t.bootstrap.unreachable_high == 0

    def test_the_counts_never_exceed_the_resamples(self) -> None:
        fit = _fit([0.0, 0.1, 0.5, 2.5], [20, 11, 9, 9], [20] * 4, model="isotonic")

        t = _boot(fit, "cell_bootstrap", 999, seed=1)

        assert t.bootstrap is not None
        b = t.bootstrap
        assert b.failed + b.unreachable_low + b.unreachable_high <= b.n_resamples
        assert b.failed == 0  # an isotonic refit cannot fail


@pytest.mark.parametrize("method", ["parametric_bootstrap", "cell_bootstrap"])
def test_a_censored_point_estimate_is_returned_before_any_resampling(method: str) -> None:
    fit = fit_dose_response(
        Observations.from_counts(
            _LOG,
            severity=[0.0, 0.5, 1.0, 2.0, 4.0],
            successes=[200] * 5,
            trials=[200] * 5,
            outcome="success",
            direction="decreasing",
        ),
        model="binomial",
        link="probit",
        upper=1.0,
        lower=0.0,
    )

    t = _boot(fit, method, 99, seed=1)

    assert t.censoring is Censoring.RIGHT and t.value is None
    assert t.bootstrap is None
