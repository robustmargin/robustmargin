"""Coverage-closing tests for the estimated-asymptote profile path in
``robustmargin.intervals``: ``_MuSubstitutedLogLikelihood``, ``_optimizer_start_from_fit``,
``_estimated_profile_llf`` and ``_profile_interval_estimated`` (see ``intervals.py``'s own
docstring on why the offset trick does not apply once an asymptote moves).

This is the path a design with a chance floor needs -- a consistency rate with a chance floor
means ``upper="estimate"`` -- and no other test reaches it: every other ``upper="estimate"``
case in the suite built from the pooled-counts fixture is ``system_b_axis_1``, which is
``Status.SEPARATION`` and short-circuits into the exact-bound fallback before any interval is
ever computed.

Uses the committed ``tools/drc_reference/estimated_asymptotes.json`` simulation (true ``mu=0,
s=0.5, upper=0.9, lower=0.1``, an 11-point log design with an exact ``severity=0`` control,
``n=400``) -- the same fixture ``tests/reference/test_models_drc.py::TestEstimatedAsymptoteParity``
reads for point-estimate parity. This file is about the *interval* path built on top of that
already-parity-checked fit, not a second copy of that parity check, so it does not re-assert
``fit.params`` against drc.

**A note on what "contains the true threshold" means here.** ``absolute(0.5)`` and
``relative(0.5))`` share a true crossing of exactly ``exp(true_mu) = 1.0`` for this simulation,
but only because ``(true_upper + true_lower) / 2 == (0.9 + 0.1) / 2 == 0.5`` happens to equal
the ``absolute`` target -- ``relative(p)``'s target is always the fitted (or true) dynamic
range's own midpoint at ``p=0.5``, regardless of what the asymptotes are, while ``absolute(0.5)``
would not coincide with it for a different true ``(upper, lower)``. ``baseline_fraction(0.5)``'s
true target is ``0.5 * true_upper = 0.45``, a different value with its own true crossing
(``~1.0818``, recomputed independently below); it is checked separately, not against the same
``1.0``.

On the deliberately-misspecified ``upper="estimate", lower=0.0`` fit (a caller-chosen fixed
floor that does not match the simulation's true ``lower=0.1`` -- the same
``upper="estimate", lower=0.0`` shape ``tools/drc_reference/README.md`` documents for drc's
``LN.3`` as "not a recovery-of-truth check"), only ``absolute(0.5)`` is checked for true-value
containment. ``baseline_fraction``/``relative`` on that fit solve for a target defined by the
*wrong* floor, so their point estimate is expected to be biased away from the true crossing --
asserting containment there would be asserting a property the model has no reason to have,
confirmed empirically before writing this file (the true ``1.0`` sits outside both intervals).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from scipy.stats import norm

from robustmargin import (
    Axis,
    Censoring,
    Definition,
    Fit,
    Observations,
    Status,
    fit_dose_response,
    threshold,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tools" / "drc_reference"
_LEVEL = 0.95

_TRUE_MU = 0.0
_TRUE_S = 0.5
_TRUE_UPPER = 0.9
_TRUE_LOWER = 0.1
# relative(0.5)'s and absolute(0.5)'s true crossings coincide here only because the true
# asymptotes' own midpoint happens to equal 0.5 (module docstring).
_TRUE_THRESHOLD_MIDPOINT = math.exp(_TRUE_MU)
# baseline_fraction(0.5)'s true target is 0.5 * true_upper = 0.45, not the midpoint -- its own
# true crossing, recomputed the same way _expected_value below does, from the true parameters.
_TRUE_TARGET_BASELINE_FRACTION = 0.5 * _TRUE_UPPER
_TRUE_Z_BASELINE_FRACTION = float(
    norm.ppf((_TRUE_UPPER - _TRUE_TARGET_BASELINE_FRACTION) / (_TRUE_UPPER - _TRUE_LOWER))
)
_TRUE_THRESHOLD_BASELINE_FRACTION = math.exp(_TRUE_MU + _TRUE_S * _TRUE_Z_BASELINE_FRACTION)


def _load_simulation() -> dict[str, Any]:
    with (_FIXTURE_DIR / "estimated_asymptotes.json").open() as fh:
        fixture: dict[str, Any] = json.load(fh)
    simulation: dict[str, Any] = fixture["simulation"]
    return simulation


def _observations(severity: list[float], successes: list[int], trials: list[int]) -> Observations:
    axis = Axis(name="severity", unit="unit", scale="log")
    return Observations.from_counts(
        axis,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )


def _simulation_observations() -> Observations:
    sim = _load_simulation()
    return _observations(sim["severity"], sim["successes"], sim["trials"])


def _target_for(definition: Definition, *, upper: float, lower: float) -> float:
    """The target performance for a definition, reimplemented independently of
    ``robustmargin.intervals._target_for`` so :func:`_expected_value` checks the solve rather than
    restating it."""
    if definition.kind == "absolute":
        return definition.value
    if definition.kind == "baseline_fraction":
        return definition.value * upper
    return upper - definition.value * (upper - lower)  # relative


def _expected_value(fit: Fit, definition: Definition) -> float:
    """``exp(mu + s * z*)`` recomputed directly from ``fit.params`` (probit only, this fixture's
    link), independently of ``robustmargin.intervals.solve_value``."""
    assert fit.params is not None
    mu = fit.params["mu"].value
    s = fit.params["s"].value
    upper = fit.params["upper"].value
    lower = fit.params["lower"].value
    target = _target_for(definition, upper=upper, lower=lower)
    q = (upper - target) / (upper - lower)
    z_star = float(norm.ppf(q))
    return math.exp(mu + s * z_star)


@pytest.fixture(scope="module")
def both_estimated_fit() -> Fit:
    fit = fit_dose_response(
        _simulation_observations(),
        model="binomial",
        link="probit",
        upper="estimate",
        lower="estimate",
    )
    assert fit.status is Status.OK
    return fit


@pytest.fixture(scope="module")
def upper_only_fit() -> Fit:
    """``upper="estimate", lower=0.0`` -- deliberately misspecified relative to the simulation's
    true ``lower=0.1`` (module docstring)."""
    fit = fit_dose_response(
        _simulation_observations(), model="binomial", link="probit", upper="estimate", lower=0.0
    )
    assert fit.status is Status.OK
    return fit


_DEFINITIONS = [
    Definition.absolute(0.5),
    Definition.baseline_fraction(0.5),
    Definition.relative(0.5),
]
_DEFINITION_IDS = ["absolute", "baseline_fraction", "relative"]


class TestBothAsymptotesEstimated:
    """``upper="estimate", lower="estimate"`` -- the fully estimated-asymptote path, correctly
    specified relative to the simulation's true asymptotes."""

    @pytest.mark.parametrize("definition", _DEFINITIONS, ids=_DEFINITION_IDS)
    @pytest.mark.parametrize("interval_method", ["profile", "delta"])
    def test_ok_none_with_value_matching_the_independently_recomputed_solve(
        self, both_estimated_fit: Fit, definition: Definition, interval_method: str
    ) -> None:
        t = threshold(
            both_estimated_fit,
            definition=definition,
            interval_method=interval_method,
            level=_LEVEL,
        )

        assert t.status is Status.OK
        assert t.censoring is Censoring.NONE
        assert t.value is not None and t.lo is not None and t.hi is not None
        assert t.lo <= t.value <= t.hi
        assert t.value == pytest.approx(_expected_value(both_estimated_fit, definition), rel=1e-6)

    @pytest.mark.parametrize("interval_method", ["profile", "delta"])
    def test_absolute_0_5_interval_contains_the_simulated_true_threshold(
        self, both_estimated_fit: Fit, interval_method: str
    ) -> None:
        t = threshold(
            both_estimated_fit,
            definition=Definition.absolute(0.5),
            interval_method=interval_method,
            level=_LEVEL,
        )
        assert t.lo is not None and t.hi is not None
        assert t.lo <= _TRUE_THRESHOLD_MIDPOINT <= t.hi

    @pytest.mark.parametrize("interval_method", ["profile", "delta"])
    def test_relative_0_5_interval_contains_the_simulated_true_threshold(
        self, both_estimated_fit: Fit, interval_method: str
    ) -> None:
        t = threshold(
            both_estimated_fit,
            definition=Definition.relative(0.5),
            interval_method=interval_method,
            level=_LEVEL,
        )
        assert t.lo is not None and t.hi is not None
        assert t.lo <= _TRUE_THRESHOLD_MIDPOINT <= t.hi

    @pytest.mark.parametrize("interval_method", ["profile", "delta"])
    def test_baseline_fraction_0_5_interval_contains_its_own_simulated_true_threshold(
        self, both_estimated_fit: Fit, interval_method: str
    ) -> None:
        """``baseline_fraction(0.5)``'s true crossing (``~1.0818``) differs from the other two
        definitions' (module docstring) -- checked against its own recomputed true value, not
        ``_TRUE_THRESHOLD_MIDPOINT``."""
        t = threshold(
            both_estimated_fit,
            definition=Definition.baseline_fraction(0.5),
            interval_method=interval_method,
            level=_LEVEL,
        )
        assert t.lo is not None and t.hi is not None
        assert t.lo <= _TRUE_THRESHOLD_BASELINE_FRACTION <= t.hi

    @pytest.mark.parametrize("definition", _DEFINITIONS, ids=_DEFINITION_IDS)
    def test_profile_and_delta_agree_closely(
        self, both_estimated_fit: Fit, definition: Definition
    ) -> None:
        """A cross-check, not a parity requirement (this module's own docstring: delta is the
        cross-check, profile is primary) -- ``rel=1e-2`` is loose on purpose. If these ever
        diverge badly, one of the two methods is wrong."""
        profile = threshold(
            both_estimated_fit, definition=definition, interval_method="profile", level=_LEVEL
        )
        delta = threshold(
            both_estimated_fit, definition=definition, interval_method="delta", level=_LEVEL
        )

        assert profile.value == pytest.approx(delta.value, rel=1e-9)
        assert profile.lo is not None and delta.lo is not None
        assert profile.hi is not None and delta.hi is not None
        assert profile.lo == pytest.approx(delta.lo, rel=1e-2)
        assert profile.hi == pytest.approx(delta.hi, rel=1e-2)


class TestUpperEstimatedLowerFixedAtZero:
    """``upper="estimate", lower=0.0`` -- the one-sided estimated-asymptote path, and
    deliberately misspecified relative to the simulation's true ``lower=0.1`` (module
    docstring): only ``absolute`` is checked for true-value containment here."""

    @pytest.mark.parametrize("definition", _DEFINITIONS, ids=_DEFINITION_IDS)
    @pytest.mark.parametrize("interval_method", ["profile", "delta"])
    def test_ok_none_with_value_matching_the_independently_recomputed_solve(
        self, upper_only_fit: Fit, definition: Definition, interval_method: str
    ) -> None:
        t = threshold(
            upper_only_fit, definition=definition, interval_method=interval_method, level=_LEVEL
        )

        assert t.status is Status.OK
        assert t.censoring is Censoring.NONE
        assert t.value is not None and t.lo is not None and t.hi is not None
        assert t.lo <= t.value <= t.hi
        assert t.value == pytest.approx(_expected_value(upper_only_fit, definition), rel=1e-6)

    @pytest.mark.parametrize("interval_method", ["profile", "delta"])
    def test_absolute_0_5_interval_contains_the_simulated_true_threshold(
        self, upper_only_fit: Fit, interval_method: str
    ) -> None:
        t = threshold(
            upper_only_fit,
            definition=Definition.absolute(0.5),
            interval_method=interval_method,
            level=_LEVEL,
        )
        assert t.lo is not None and t.hi is not None
        assert t.lo <= _TRUE_THRESHOLD_MIDPOINT <= t.hi

    def test_baseline_fraction_and_relative_coincide_when_lower_is_fixed_at_zero(
        self, upper_only_fit: Fit
    ) -> None:
        """``baseline_fraction(p)``'s target is ``p * upper``; ``relative(p)``'s is
        ``upper - p * (upper - lower)``. The two are algebraically identical whenever
        ``lower == 0`` -- true here by construction (the fixed ``lower=0.0`` argument), not a
        coincidence of the data -- so both definitions must produce the identical threshold."""
        baseline = threshold(
            upper_only_fit,
            definition=Definition.baseline_fraction(0.5),
            interval_method="profile",
            level=_LEVEL,
        )
        relative = threshold(
            upper_only_fit,
            definition=Definition.relative(0.5),
            interval_method="profile",
            level=_LEVEL,
        )

        assert baseline.value == relative.value
        assert baseline.lo == relative.lo
        assert baseline.hi == relative.hi

    @pytest.mark.parametrize("definition", _DEFINITIONS, ids=_DEFINITION_IDS)
    def test_profile_and_delta_agree_closely(
        self, upper_only_fit: Fit, definition: Definition
    ) -> None:
        profile = threshold(
            upper_only_fit, definition=definition, interval_method="profile", level=_LEVEL
        )
        delta = threshold(
            upper_only_fit, definition=definition, interval_method="delta", level=_LEVEL
        )

        assert profile.value == pytest.approx(delta.value, rel=1e-9)
        assert profile.lo is not None and delta.lo is not None
        assert profile.hi is not None and delta.hi is not None
        assert profile.lo == pytest.approx(delta.lo, rel=1e-2)
        assert profile.hi == pytest.approx(delta.hi, rel=1e-2)


class TestSparseEstimatedFitOpensItsLowerSide:
    """A verified ``OPEN_LOWER`` on the estimated-asymptote path, on a sparse design.

    Subsampling the committed 11-level simulation to ``n=12`` per level (successes rounded
    proportionally from the ``n=400`` counts -- deterministic, no random draw, reproducible
    bit-for-bit) gives a genuinely sparse design. Under ``absolute(0.85)`` the upper asymptote
    can fall to the target, so the constrained maximum on the lower side tends to the
    log-likelihood of the fit with the upper asymptote fixed at 0.85 (the wall). The wall
    deviance 2 * (llf_hat - llf_wall) is 1.026, inside the chi-square(1) target of 3.84. That is
    necessary for the lower side to be open, not sufficient on its own: it shows the limit the
    profile tends to stays below the target. That the side is open is what the profile test
    asserts. The upper side still closes.
    """

    @staticmethod
    def _sparse_fit() -> Fit:
        sim = _load_simulation()
        n = 12
        successes = [round(s * n / t) for s, t in zip(sim["successes"], sim["trials"], strict=True)]
        trials = [n] * len(sim["severity"])
        fit = fit_dose_response(
            _observations(sim["severity"], successes, trials),
            model="binomial",
            link="probit",
            upper="estimate",
            lower="estimate",
        )
        assert fit.status is Status.OK
        return fit

    def test_absolute_0_85_profile_opens_the_lower_side_and_closes_the_upper_side(self) -> None:
        """The estimated-asymptote profile reports ``OPEN_LOWER`` with ``lo is None`` and a
        closed ``hi`` above the point estimate.

        Oracle: ``fit_dose_response`` with the upper asymptote fixed at the target (0.85), a
        different fit path from the profile code, has llf -13.052007 against -12.538999 for the
        full fit. The deviance, twice that difference, is 1.026, below the chi-square(1) 0.95
        quantile (3.841), so the profile never reaches the target on the lower side.
        """
        fit = self._sparse_fit()

        t = threshold(
            fit, definition=Definition.absolute(0.85), interval_method="profile", level=_LEVEL
        )

        assert t.status is Status.OK
        assert t.censoring is Censoring.OPEN_LOWER
        assert t.lo is None
        assert t.value is not None
        assert t.hi is not None
        assert t.value < t.hi

    def test_wall_fit_deviance_is_inside_the_chi_square_target(self) -> None:
        """Oracle behind the open lower side: the fit with the upper asymptote fixed at 0.85
        loses 1.026 (twice the llf difference) against the full fit, below the 3.841 target.
        Necessary for the lower side to be open, not sufficient on its own: it shows the wall
        limit the profile tends to stays below the target. The open side itself is asserted by
        the profile test above."""
        fit = self._sparse_fit()
        sim = _load_simulation()
        n = 12
        successes = [round(s * n / t) for s, t in zip(sim["successes"], sim["trials"], strict=True)]
        wall = fit_dose_response(
            _observations(sim["severity"], successes, [n] * len(successes)),
            model="binomial",
            link="probit",
            upper=0.85,
            lower="estimate",
        )

        assert fit.log_likelihood is not None and wall.log_likelihood is not None
        assert fit.log_likelihood == pytest.approx(-12.538999, abs=1e-5)
        assert wall.log_likelihood == pytest.approx(-13.052007, abs=1e-5)
        deviance = 2.0 * (fit.log_likelihood - wall.log_likelihood)
        assert deviance == pytest.approx(1.026, abs=1e-3)
        assert deviance < 3.841

    def test_delta_still_gives_a_closed_interval_on_the_identical_fit_and_definition(
        self,
    ) -> None:
        """``IntervalResult``'s own docstring: the delta method has no open-sided outcome.
        Cross-checked on the same sparse fit and definition as the profile above: the delta
        method closes both sides where the profile opens the lower one, the expected shape at
        small ``n`` where a Wald-type interval is the less trustworthy one."""
        fit = self._sparse_fit()

        t = threshold(
            fit, definition=Definition.absolute(0.85), interval_method="delta", level=_LEVEL
        )

        assert t.status is Status.OK
        assert t.censoring is Censoring.NONE
        assert t.lo is not None and t.hi is not None
