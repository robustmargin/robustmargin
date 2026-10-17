"""Unit tests for the data-only monotonicity check: :class:`~robustmargin.MonotonicityCheck`,
:class:`~robustmargin.AdjacentPair` and :func:`~robustmargin.check_monotonicity`.

The check lives on ``Fit.monotonicity`` directly, not on ``Diagnostics`` (which needs a
converged fit), because it needs only raw per-level counts -- it must still run when the curve
fit fails, which is exactly when non-monotone data is most often the cause. Every expected
number here is computed independently, using the exact formula: a one-sided
``scipy.stats.fisher_exact`` test on
``[[successes_high, failures_high], [successes_low, failures_low]]`` with
``alternative="greater"`` for ``direction="decreasing"`` (``"less"`` for ``"increasing"``), Holm
across all pairs. A from-scratch Holm step-down is kept alongside ``statsmodels``'s
``multipletests(method="holm")`` as a second, independent cross-check (worth keeping even
though ``statsmodels`` is the implementation).
"""

from __future__ import annotations

import dataclasses

import pytest
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from robustmargin import (
    AdjacentPair,
    Axis,
    Definition,
    MonotonicityCheck,
    Observations,
    Status,
    check_monotonicity,
    fit_dose_response,
    threshold,
)

_LOG_AXIS = Axis(name="severity", unit="unit", scale="log")


def _obs(
    severity: list[float],
    successes: list[int],
    trials: list[int],
    *,
    direction: str = "decreasing",
    outcome: str = "success",
    cluster: list[str] | None = None,
    axis: Axis = _LOG_AXIS,
) -> Observations:
    return Observations.from_counts(
        axis,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome=outcome,
        direction=direction,
        cluster=cluster,
    )


def _holm_step_down(p_values: list[float]) -> list[float]:
    """A direct, independent Holm step-down (Holm 1979): sort ascending, multiply by the
    remaining count, take the running maximum, clip at 1. Deliberately not ``statsmodels``'s
    ``multipletests`` -- kept as a from-scratch cross-check alongside it."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        factor = m - rank
        running_max = max(running_max, min(p_values[idx] * factor, 1.0))
        adjusted[idx] = running_max
    return adjusted


def _expected_raw_p_values(levels: list[tuple[int, int]], *, direction: str) -> list[float]:
    """``levels`` is ``[(successes, trials), ...]`` sorted ascending by severity, matching the
    exact table/alternative convention."""
    alternative = "greater" if direction == "decreasing" else "less"
    raw = []
    for (s_lo, n_lo), (s_hi, n_hi) in zip(levels, levels[1:], strict=False):
        f_lo, f_hi = n_lo - s_lo, n_hi - s_hi
        table = [[s_hi, f_hi], [s_lo, f_lo]]
        _, p = fisher_exact(table, alternative=alternative)
        raw.append(float(p))
    return raw


# ------------------------------------------------------------------------------------------
# Holm adjustment: raw p-values match the exact Fisher exact formula; the
# Holm-adjusted p-values match both statsmodels and a from-scratch step-down.
# ------------------------------------------------------------------------------------------


class TestHolmAdjustment:
    """Four severities, three adjacent pairs: a genuine reversal at the first pair (57/200-style
    magnitude), then a sharp, unambiguous decline, then a tie -- so Holm has real work to do
    (only the first pair should survive correction)."""

    _SEVERITY = [0.0, 1.0, 2.0, 3.0]
    _SUCCESSES = [40, 120, 5, 2]
    _TRIALS = [200, 300, 300, 300]

    def _obs(self) -> Observations:
        return _obs(self._SEVERITY, self._SUCCESSES, self._TRIALS)

    def _levels(self) -> list[tuple[int, int]]:
        return list(zip(self._SUCCESSES, self._TRIALS, strict=True))

    def test_raw_p_values_match_the_amendments_exact_fisher_exact_formula(self) -> None:
        expected_raw = _expected_raw_p_values(self._levels(), direction="decreasing")

        check = check_monotonicity(self._obs())

        assert [pair.p_value for pair in check.pairs] == pytest.approx(expected_raw, rel=1e-9)

    def test_holm_adjusted_p_values_match_statsmodels(self) -> None:
        expected_raw = _expected_raw_p_values(self._levels(), direction="decreasing")
        _, expected_holm, _, _ = multipletests(expected_raw, method="holm")

        check = check_monotonicity(self._obs())

        assert [pair.holm_adjusted_p for pair in check.pairs] == pytest.approx(
            list(expected_holm), rel=1e-9
        )

    def test_holm_adjusted_p_values_also_match_an_independent_step_down(self) -> None:
        expected_raw = _expected_raw_p_values(self._levels(), direction="decreasing")
        expected_holm = _holm_step_down(expected_raw)

        check = check_monotonicity(self._obs())

        assert [pair.holm_adjusted_p for pair in check.pairs] == pytest.approx(
            expected_holm, rel=1e-9
        )
        assert [pair.flagged for pair in check.pairs] == [h < 0.05 for h in expected_holm]

    def test_only_the_genuine_reversal_is_flagged(self) -> None:
        check = check_monotonicity(self._obs())

        flagged = [(p.severity_low, p.severity_high) for p in check.pairs if p.flagged]

        assert flagged == [(0.0, 1.0)]
        assert check.any_flagged is True

    def test_pairs_are_in_ascending_severity_order(self) -> None:
        check = check_monotonicity(self._obs())

        assert [(p.severity_low, p.severity_high) for p in check.pairs] == [
            (0.0, 1.0),
            (1.0, 2.0),
            (2.0, 3.0),
        ]

    def test_pairs_carry_the_raw_counts_alongside_the_p_values(self) -> None:
        check = check_monotonicity(self._obs())

        first = check.pairs[0]
        assert (first.successes_low, first.trials_low) == (40, 200)
        assert (first.successes_high, first.trials_high) == (120, 300)

    def test_every_pair_is_an_adjacent_pair_instance(self) -> None:
        check = check_monotonicity(self._obs())

        assert all(isinstance(pair, AdjacentPair) for pair in check.pairs)

    def test_check_records_direction_alpha_dependence_and_method(self) -> None:
        check = check_monotonicity(self._obs())

        assert check.direction == "decreasing"
        assert check.alpha == 0.05
        assert check.dependence == "independent"
        assert check.method == "fisher_exact_one_sided_holm"

    def test_custom_alpha_is_recorded_and_changes_flagging(self) -> None:
        """A dataset chosen so the first pair's Holm-adjusted p (~0.245) sits strictly between
        the two alphas compared -- unlike ``TestHolmAdjustment``'s own dataset, whose
        holm-adjusted p-values are either ~0 or exactly 1.0 and so cannot distinguish "alpha is
        actually used" from "alpha is silently ignored and hardcoded at 0.05"."""
        obs = _obs([0.0, 1.0, 2.0], [3, 7, 1], [15, 15, 15])

        default = check_monotonicity(obs, alpha=0.05)
        loose = check_monotonicity(obs, alpha=0.3)

        assert default.alpha == 0.05
        assert loose.alpha == 0.3
        assert [p.flagged for p in default.pairs] == [False, False]
        assert [p.flagged for p in loose.pairs] == [True, False]


# ------------------------------------------------------------------------------------------
# The control -> first-level pair is in the family even when the fitted likelihood (fixed
# u=1, l=0) drops the control row entirely -- monotonicity reads
# the raw Observations, never the likelihood-contributing subset.
# ------------------------------------------------------------------------------------------


class TestControlPairIsAlwaysIncluded:
    def test_control_to_first_level_pair_is_present_via_check_monotonicity(self) -> None:
        obs = _obs([0.0, 1.0, 2.0], [50, 30, 10], [50, 50, 50])

        check = check_monotonicity(obs)

        assert (0.0, 1.0) in [(p.severity_low, p.severity_high) for p in check.pairs]

    def test_control_pair_survives_the_fixed_u_1_l_0_fit_too(self) -> None:
        obs = _obs([0.0, 1.0, 2.0], [50, 30, 10], [50, 50, 50])

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        assert fit.status is Status.OK
        assert fit.monotonicity is not None
        assert (0.0, 1.0) in [(p.severity_low, p.severity_high) for p in fit.monotonicity.pairs]


# ------------------------------------------------------------------------------------------
# Direction symmetry: mirroring outcome (success <-> failure) and direction (decreasing <->
# increasing) gives the same pairs with the same p-values (verified independently above to be
# an exact algebraic identity of the Fisher exact test, not merely "close").
# ------------------------------------------------------------------------------------------


class TestDirectionSymmetry:
    _SEVERITY = [0.0, 1.0, 2.0, 3.0]
    _SUCCESSES = [40, 120, 5, 2]
    _TRIALS = [200, 300, 300, 300]

    def test_mirrored_outcome_and_direction_give_the_same_pairs_and_p_values(self) -> None:
        original = check_monotonicity(
            _obs(self._SEVERITY, self._SUCCESSES, self._TRIALS, direction="decreasing")
        )
        mirrored_successes = [n - s for s, n in zip(self._SUCCESSES, self._TRIALS, strict=True)]
        mirrored = check_monotonicity(
            _obs(
                self._SEVERITY,
                mirrored_successes,
                self._TRIALS,
                direction="increasing",
                outcome="failure",
            )
        )

        def _summary(check: MonotonicityCheck) -> list[tuple[float, float, float, float, bool]]:
            return [
                (p.severity_low, p.severity_high, p.p_value, p.holm_adjusted_p, p.flagged)
                for p in check.pairs
            ]

        original_summary = _summary(original)
        mirrored_summary = _summary(mirrored)
        assert len(original_summary) == len(mirrored_summary)
        for (sl1, sh1, p1, h1, f1), (sl2, sh2, p2, h2, f2) in zip(
            original_summary, mirrored_summary, strict=True
        ):
            assert sl1 == sl2
            assert sh1 == sh2
            assert p1 == pytest.approx(p2, rel=1e-9)
            assert h1 == pytest.approx(h2, rel=1e-9)
            assert f1 == f2


# ------------------------------------------------------------------------------------------
# Clustered input: cluster ids without dependence="independent" raise, both for
# check_monotonicity directly and for fit_dose_response (the argument threads through).
# ------------------------------------------------------------------------------------------


class TestClusteredInputRequiresExplicitDependence:
    _SEVERITY = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
    _SUCCESSES = [1, 1, 1, 0, 0, 0]
    _TRIALS = [1, 1, 1, 1, 1, 1]
    _CLUSTER = ["a", "b", "a", "b", "a", "b"]

    def _clustered_obs(self) -> Observations:
        return _obs(self._SEVERITY, self._SUCCESSES, self._TRIALS, cluster=self._CLUSTER)

    def test_check_monotonicity_raises_without_dependence(self) -> None:
        with pytest.raises(ValueError):
            check_monotonicity(self._clustered_obs())

    def test_check_monotonicity_accepts_explicit_independent(self) -> None:
        check = check_monotonicity(self._clustered_obs(), dependence="independent")

        assert check.dependence == "independent"

    def test_fit_dose_response_raises_on_clustered_input_without_dependence(self) -> None:
        with pytest.raises(ValueError):
            fit_dose_response(
                self._clustered_obs(), model="binomial", link="probit", upper=1.0, lower=0.0
            )

    def test_fit_dose_response_accepts_explicit_independent_dependence(self) -> None:
        fit = fit_dose_response(
            self._clustered_obs(),
            model="binomial",
            link="probit",
            upper=1.0,
            lower=0.0,
            dependence="independent",
        )

        assert fit.monotonicity is not None
        assert fit.monotonicity.dependence == "independent"

    def test_unclustered_input_needs_no_dependence_argument(self) -> None:
        unclustered = _obs([0.0, 1.0, 2.0], [10, 5, 1], [10, 10, 10])

        check = check_monotonicity(unclustered)

        assert check.dependence == "independent"


# ------------------------------------------------------------------------------------------
# Degenerate: fewer than two distinct severity levels gives an empty family, not an error.
# ------------------------------------------------------------------------------------------


class TestDegenerateSingleLevel:
    def test_a_single_level_gives_empty_pairs_and_is_not_flagged(self) -> None:
        obs = _obs([1.0], [5], [10])

        check = check_monotonicity(obs)

        assert check.pairs == ()
        assert check.any_flagged is False
        assert check.warnings
        assert any("level" in w.lower() for w in check.warnings)

    def test_repeated_rows_at_one_severity_still_aggregate_to_a_single_level(self) -> None:
        """Many rows at the *same* severity still pool to one level, not one pair per row."""
        obs = _obs([1.0, 1.0, 1.0], [3, 1, 4], [5, 5, 5])

        check = check_monotonicity(obs)

        assert check.pairs == ()
        assert check.any_flagged is False


# ------------------------------------------------------------------------------------------
# Threshold invariance: threshold() numbers and statuses are identical whether fit.monotonicity
# is flagged or not -- only warnings differs (nothing about the reported threshold changes; a
# flagged check only adds a warning token).
# ------------------------------------------------------------------------------------------


def _unflagged_check() -> MonotonicityCheck:
    return MonotonicityCheck(
        direction="decreasing",
        alpha=0.05,
        dependence="independent",
        method="fisher_exact_one_sided_holm",
        pairs=(
            AdjacentPair(
                severity_low=0.0,
                severity_high=1.0,
                successes_low=90,
                trials_low=100,
                successes_high=80,
                trials_high=100,
                p_value=0.9,
                holm_adjusted_p=0.9,
                flagged=False,
            ),
        ),
        any_flagged=False,
        warnings=(),
    )


def _flagged_check() -> MonotonicityCheck:
    return dataclasses.replace(
        _unflagged_check(),
        pairs=(
            dataclasses.replace(
                _unflagged_check().pairs[0],
                p_value=1e-6,
                holm_adjusted_p=1e-6,
                flagged=True,
            ),
        ),
        any_flagged=True,
    )


class TestThresholdInvarianceUnderMonotonicityCheckSubstitution:
    def _base_fit(self):
        axis = Axis(name="severity", unit="unit", scale="log")
        obs = Observations.from_counts(
            axis,
            severity=[0.0, 1.0, 2.0, 4.0],
            successes=[100, 90, 55, 5],
            trials=[100, 100, 100, 100],
            outcome="success",
            direction="decreasing",
        )
        return fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

    def test_only_warnings_differ_between_a_flagged_and_unflagged_check(self) -> None:
        fit = self._base_fit()
        assert fit.status is Status.OK

        fit_unflagged = dataclasses.replace(fit, monotonicity=_unflagged_check())
        fit_flagged = dataclasses.replace(fit, monotonicity=_flagged_check())

        t_unflagged = threshold(
            fit_unflagged,
            definition=Definition.absolute(0.5),
            interval_method="profile",
            level=0.95,
        )
        t_flagged = threshold(
            fit_flagged,
            definition=Definition.absolute(0.5),
            interval_method="profile",
            level=0.95,
        )

        assert t_unflagged.value == t_flagged.value
        assert t_unflagged.lo == t_flagged.lo
        assert t_unflagged.hi == t_flagged.hi
        assert t_unflagged.level == t_flagged.level
        assert t_unflagged.interval_method == t_flagged.interval_method
        assert t_unflagged.censoring == t_flagged.censoring
        assert t_unflagged.status == t_flagged.status
        assert t_unflagged.dependence == t_flagged.dependence
        assert t_unflagged.definition == t_flagged.definition
        assert t_unflagged.axis == t_flagged.axis
        assert t_unflagged.direction == t_flagged.direction
        assert t_unflagged.baseline == t_flagged.baseline
        assert t_unflagged.grid == t_flagged.grid

        assert not any(w.startswith("NON_MONOTONE_DATA: ") for w in t_unflagged.warnings)
        assert any(w.startswith("NON_MONOTONE_DATA: ") for w in t_flagged.warnings)
