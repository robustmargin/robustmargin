"""Property tests for ``threshold()`` (hypothesis).

Mirrors ``tests/property/test_models_properties.py``'s conventions: small examples so real
``fit_dose_response``/``threshold`` calls stay fast, ``hypothesis.assume`` to skip examples that
do not converge to an interior, closed threshold (the headline properties below are about what
a **successful, uncensored** threshold does), and no hidden randomness beyond hypothesis's own
(seeded, deterministic) search.

Five properties:

- **Scale invariance (the headline unit-correctness property: unit correctness is a
  correctness property).** Rescaling ``x -> c*x`` scales ``value``, ``lo`` and ``hi`` by
  exactly ``c``, on both a log and a linear axis, for both ``interval_method="profile"`` and
  ``"delta"``.
- **Row permutation** changes nothing.
- **Aggregated and disaggregated inputs** give the same threshold, following directly from the
  fit-level "same fit" property.
- **With ``upper`` fixed at 1.0, adding an all-success zero-severity control** leaves the
  threshold unchanged, on a log axis (the only axis a zero-severity control means anything
  on).

The last three reduce to "the same ``Fit`` gives the same ``Threshold``" plus the fit-level
invariants proven in ``tests/property/test_models_properties.py``, so they are direct, not just
structurally analogous.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

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

_MAX_EXAMPLES = 25
_LOG_AXIS = Axis(name="severity", unit="unit", scale="log")
_LINEAR_AXIS = Axis(name="severity", unit="unit", scale="linear")
_LEVEL = 0.95
_RESCALE_FACTOR = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)


@st.composite
def _non_separated_counts(draw: st.DrawFn) -> tuple[list[float], list[int], list[int]]:
    """A counts table with **every** level carrying at least one success and one failure, which
    provably never separates (the pre-fit check): see
    ``tests/property/test_models_properties.py`` for the argument, reused verbatim here.
    """
    n = draw(st.integers(min_value=2, max_value=4))
    severities = draw(
        st.lists(
            st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    trials = draw(st.lists(st.integers(min_value=10, max_value=40), min_size=n, max_size=n))
    successes = [draw(st.integers(min_value=1, max_value=t - 1)) for t in trials]
    return severities, successes, trials


@st.composite
def _informative_counts(draw: st.DrawFn) -> tuple[list[float], list[int], list[int]]:
    """A counts table that actually constrains a threshold: a monotonically decreasing success
    rate straddling 0.5, at enough trials per level for the profile to close.

    ``_non_separated_counts`` draws each level's successes independently and uniformly at 10-40
    trials, so most of what it produces is a shallow, non-monotone curve whose profile deviance
    never crosses the chi-square target -- a real result but a vacuous input
    for a scale-invariance property, and enough of them to trip hypothesis's filter health
    check. This generator keeps the same no-separation guarantee (every level keeps at least one
    success and one failure) while producing designs where a closed interval exists.
    """
    n = draw(st.integers(min_value=3, max_value=4))
    severities = sorted(
        draw(
            st.lists(
                st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
                unique=True,
            )
        )
    )
    trials = draw(st.lists(st.integers(min_value=80, max_value=200), min_size=n, max_size=n))
    top = draw(st.floats(min_value=0.80, max_value=0.95))
    bottom = draw(st.floats(min_value=0.05, max_value=0.20))
    step = (top - bottom) / (n - 1)
    successes = [
        min(max(int(round((top - i * step) * n_trials)), 1), n_trials - 1)
        for i, n_trials in enumerate(trials)
    ]
    return severities, successes, trials


def _from_counts(counts: tuple[list[float], list[int], list[int]], *, axis: Axis) -> Observations:
    severity, successes, trials = counts
    return Observations.from_counts(
        axis,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )


def _rows_from_counts(
    counts: tuple[list[float], list[int], list[int]], *, axis: Axis
) -> Observations:
    severity, successes, trials = counts
    row_severity: list[float] = []
    row_success: list[int] = []
    for level, n_success, n_trials in zip(severity, successes, trials, strict=True):
        row_severity.extend([level] * n_trials)
        row_success.extend([1] * n_success + [0] * (n_trials - n_success))
    return Observations.from_observations(
        axis,
        severity=row_severity,
        success=row_success,
        outcome="success",
        direction="decreasing",
    )


def _fit_fixed_trivial(obs: Observations) -> Fit:
    return fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)


def _closed_threshold_or_none(fit: Fit, *, interval_method: str) -> Threshold | None:
    """``absolute(0.5)`` is the natural target for ``u=1, l=0`` data (the curve's own midpoint,
    ``Fit``'s docstring). Returns ``None`` when the fit did not converge or the resulting
    threshold is not an interior, closed ("NONE" censoring, ``OK`` status) result -- the caller
    is expected to ``assume()`` on that.
    """
    if fit.status is not Status.OK:
        return None
    t = threshold(
        fit, definition=Definition.absolute(0.5), interval_method=interval_method, level=_LEVEL
    )
    if t.status is not Status.OK or t.censoring is not Censoring.NONE:
        return None
    if t.lo is None or t.hi is None:
        # (OK, NONE) does not imply finite bounds. A design too coarse to
        # constrain the threshold reports the estimate with no interval, which is a real result
        # but not a *closed* one, so it is not what these scale-invariance properties are about.
        return None
    return t


@pytest.mark.parametrize("interval_method", ["profile", "delta"])
@given(_informative_counts(), _RESCALE_FACTOR)
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_rescaling_severity_scales_value_lo_hi_by_c_on_a_log_axis(
    interval_method: str,
    counts: tuple[list[float], list[int], list[int]],
    c: float,
) -> None:
    severity, successes, trials = counts
    rescaled = ([c * s for s in severity], successes, trials)

    fit_original = _fit_fixed_trivial(_from_counts(counts, axis=_LOG_AXIS))
    fit_rescaled = _fit_fixed_trivial(_from_counts(rescaled, axis=_LOG_AXIS))

    t_original = _closed_threshold_or_none(fit_original, interval_method=interval_method)
    t_rescaled = _closed_threshold_or_none(fit_rescaled, interval_method=interval_method)
    assume(t_original is not None)
    assume(t_rescaled is not None)
    assert t_original is not None and t_rescaled is not None  # narrows for mypy-style readers

    for field in ("value", "lo", "hi"):
        original_value = getattr(t_original, field)
        rescaled_value = getattr(t_rescaled, field)
        assert rescaled_value == pytest.approx(c * original_value, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("interval_method", ["profile", "delta"])
@given(_informative_counts(), _RESCALE_FACTOR)
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_rescaling_severity_scales_value_lo_hi_by_c_on_a_linear_axis(
    interval_method: str,
    counts: tuple[list[float], list[int], list[int]],
    c: float,
) -> None:
    severity, successes, trials = counts
    rescaled = ([c * s for s in severity], successes, trials)

    fit_original = _fit_fixed_trivial(_from_counts(counts, axis=_LINEAR_AXIS))
    fit_rescaled = _fit_fixed_trivial(_from_counts(rescaled, axis=_LINEAR_AXIS))

    t_original = _closed_threshold_or_none(fit_original, interval_method=interval_method)
    t_rescaled = _closed_threshold_or_none(fit_rescaled, interval_method=interval_method)
    assume(t_original is not None)
    assume(t_rescaled is not None)
    assert t_original is not None and t_rescaled is not None

    for field in ("value", "lo", "hi"):
        original_value = getattr(t_original, field)
        rescaled_value = getattr(t_rescaled, field)
        assert rescaled_value == pytest.approx(c * original_value, rel=1e-6, abs=1e-9)


@given(_non_separated_counts(), st.data())
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_row_permutation_does_not_change_the_threshold(
    counts: tuple[list[float], list[int], list[int]],
    data: st.DataObject,
) -> None:
    severity, successes, trials = counts
    n = len(severity)
    order = data.draw(st.permutations(range(n)))
    permuted = (
        [severity[i] for i in order],
        [successes[i] for i in order],
        [trials[i] for i in order],
    )

    fit_original = _fit_fixed_trivial(_from_counts(counts, axis=_LOG_AXIS))
    fit_permuted = _fit_fixed_trivial(_from_counts(permuted, axis=_LOG_AXIS))
    assume(fit_original.status is Status.OK)
    assume(fit_permuted.status is Status.OK)

    t_original = threshold(
        fit_original, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL
    )
    t_permuted = threshold(
        fit_permuted, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL
    )

    assert t_original.status == t_permuted.status
    assert t_original.censoring == t_permuted.censoring
    assert t_original.value == t_permuted.value
    assert t_original.lo == t_permuted.lo
    assert t_original.hi == t_permuted.hi


@given(_non_separated_counts())
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_aggregated_and_disaggregated_inputs_give_the_same_threshold(
    counts: tuple[list[float], list[int], list[int]],
) -> None:
    fit_aggregated = _fit_fixed_trivial(_from_counts(counts, axis=_LOG_AXIS))
    fit_disaggregated = _fit_fixed_trivial(_rows_from_counts(counts, axis=_LOG_AXIS))
    assume(fit_aggregated.status is Status.OK)
    assume(fit_disaggregated.status is Status.OK)

    t_aggregated = threshold(
        fit_aggregated,
        definition=Definition.absolute(0.5),
        interval_method="profile",
        level=_LEVEL,
    )
    t_disaggregated = threshold(
        fit_disaggregated,
        definition=Definition.absolute(0.5),
        interval_method="profile",
        level=_LEVEL,
    )

    assert t_aggregated.status == t_disaggregated.status
    assert t_aggregated.censoring == t_disaggregated.censoring
    assert t_aggregated.value == t_disaggregated.value
    assert t_aggregated.lo == t_disaggregated.lo
    assert t_aggregated.hi == t_disaggregated.hi


@given(_non_separated_counts(), st.integers(min_value=1, max_value=200))
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_adding_an_all_success_control_leaves_a_fixed_u_1_threshold_unchanged(
    counts: tuple[list[float], list[int], list[int]],
    control_trials: int,
) -> None:
    severity, successes, trials = counts
    without_control = (severity, successes, trials)
    with_control = ([0.0, *severity], [control_trials, *successes], [control_trials, *trials])

    fit_without = _fit_fixed_trivial(_from_counts(without_control, axis=_LOG_AXIS))
    fit_with = _fit_fixed_trivial(_from_counts(with_control, axis=_LOG_AXIS))
    assume(fit_without.status is Status.OK)
    assume(fit_with.status is Status.OK)

    t_without = threshold(
        fit_without, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL
    )
    t_with = threshold(
        fit_with, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL
    )

    # The control-row property is about the *fit*, and that much holds exactly: with `u`
    # fixed at 1 an all-success control contributes `lchoose(n, n) + n*log(1) = 0` to the
    # likelihood, an identity recorded in `tools/drc_reference/README.md`.
    assert fit_without.params == fit_with.params
    assert fit_without.log_likelihood == pytest.approx(fit_with.log_likelihood, rel=1e-12)

    # The *threshold* is a weaker claim, and it is false in one direction on purpose. Section
    # 5.5's LEFT rule reads the control cell directly ("the lowest nonzero level already fails
    # while the control passes"), not through the likelihood, so a series with no control can
    # never be LEFT-censored and the same series with an all-success control can be. An
    # all-success control is likelihood-neutral but not censoring-neutral. Where the
    # classification does agree, every number must.
    if t_without.censoring is t_with.censoring and t_without.status is t_with.status:
        assert t_without.value == t_with.value
        assert t_without.lo == t_with.lo
        assert t_without.hi == t_with.hi
    else:
        assert t_with.censoring is Censoring.LEFT


@given(_informative_counts())
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_wider_confidence_level_never_gives_a_narrower_interval(
    counts: tuple[list[float], list[int], list[int]],
) -> None:
    """Interval containment/monotonicity ("a wider confidence level never produces a narrower
    interval"): 0.99 must bracket at least as widely as 0.80.
    """
    fit = _fit_fixed_trivial(_from_counts(counts, axis=_LOG_AXIS))
    assume(fit.status is Status.OK)

    t_narrow = threshold(
        fit, definition=Definition.absolute(0.5), interval_method="profile", level=0.80
    )
    t_wide = threshold(
        fit, definition=Definition.absolute(0.5), interval_method="profile", level=0.99
    )
    assume(t_narrow.status is Status.OK and t_narrow.censoring is Censoring.NONE)
    assume(t_wide.status is Status.OK and t_wide.censoring is Censoring.NONE)
    assert t_narrow.lo is not None and t_narrow.hi is not None
    assert t_wide.lo is not None and t_wide.hi is not None

    assert t_wide.lo <= t_narrow.lo
    assert t_wide.hi >= t_narrow.hi


@given(_non_separated_counts())
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_interval_containment_lo_le_value_le_hi_whenever_all_three_are_finite(
    counts: tuple[list[float], list[int], list[int]],
) -> None:
    fit = _fit_fixed_trivial(_from_counts(counts, axis=_LOG_AXIS))
    assume(fit.status is Status.OK)

    for interval_method in ("profile", "delta"):
        t = threshold(
            fit,
            definition=Definition.absolute(0.5),
            interval_method=interval_method,
            level=_LEVEL,
        )
        if t.value is not None and t.lo is not None and t.hi is not None:
            assert t.lo <= t.value <= t.hi


@given(_informative_counts())
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_a_monotone_curves_absolute_threshold_moves_monotonically_with_the_criterion(
    counts: tuple[list[float], list[int], list[int]],
) -> None:
    """Monotonicity: for a fitted curve that is decreasing by
    construction (every ``binomial`` fit is, in this release), a *lower* target performance
    solves for a *larger or equal* threshold severity -- reaching a smaller fraction of the
    dynamic range never requires *less* severity.
    """
    fit = _fit_fixed_trivial(_from_counts(counts, axis=_LOG_AXIS))
    assume(fit.status is Status.OK)

    t_high_target = threshold(
        fit, definition=Definition.absolute(0.7), interval_method="profile", level=_LEVEL
    )
    t_low_target = threshold(
        fit, definition=Definition.absolute(0.3), interval_method="profile", level=_LEVEL
    )
    assume(t_high_target.status is Status.OK and t_high_target.censoring is Censoring.NONE)
    assume(t_low_target.status is Status.OK and t_low_target.censoring is Censoring.NONE)
    assert t_high_target.value is not None and t_low_target.value is not None

    assert t_low_target.value >= t_high_target.value


def test_censored_threshold_never_has_a_finite_point_estimate() -> None:
    """Censoring never yields a finite point estimate: a ``RIGHT``-censored result has
    ``value=None`` and a labelled bound (``lo``), never a number. Uses the fixed,
    non-hypothesis ``system_a_axis_2`` worked case (complete separation) directly, since
    complete separation with all-success data is the reliable way to reach ``RIGHT`` on a
    log-or-linear axis without depending on a hypothesis search finding it by chance.
    """
    axis = Axis(name="axis_2", unit="m/s", scale="linear")
    obs = Observations.from_counts(
        axis,
        severity=[0.0, 2.0, 5.0, 10.0],
        successes=[100, 500, 500, 500],
        trials=[100, 500, 500, 500],
        outcome="success",
        direction="decreasing",
    )
    fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

    t = threshold(
        fit, definition=Definition.absolute(0.95), interval_method="profile", level=_LEVEL
    )

    assert t.censoring is Censoring.RIGHT
    assert t.value is None
    assert isinstance(t.lo, float)
    assert math.isfinite(t.lo)
