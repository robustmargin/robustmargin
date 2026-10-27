"""Direction-flip properties (hypothesis).

For binomial data, ``direction="increasing"`` is the mirror of a decreasing success rate, and no
new estimator is involved. These properties pin that: an increasing fit on events ``k`` out of
``n`` equals a decreasing fit on the complement ``n - k``, with the asymptotes mirrored,
``(u', l') = (1 - l, 1 - u)``, and with ``"estimate"`` mapping to ``"estimate"``.

What must carry over, and how it is compared:

* ``mu``, ``s``, ``log_likelihood`` and ``status`` are identical.
* ``upper = 1 - lower'`` and ``lower = 1 - upper'``, each ``fixed`` flag carried along.
* The covariance swaps the ``upper``/``lower`` names, and every entry that involves exactly one of
  the two asymptotes changes sign (the Jacobian of ``p -> 1 - p`` is ``-1``); entries that involve
  both, or neither, do not.
* ``Fit.predict`` mirrors: ``estimate = 1 - estimate'``, ``lo = 1 - hi'``, ``hi = 1 - lo'``.
* ``threshold()``: ``value``, ``lo``, ``hi``, ``status``, ``censoring``, ``interval_method`` and
  ``grid`` are identical for ``absolute(a)`` against ``absolute(1 - a)`` and for ``relative(p)``
  against ``relative(p)``. Severities are never mirrored.

Tolerance is relative ``1e-9`` throughout. Both sides run the same code on the same numbers, so
the two sides agree to rounding; ``1e-9`` leaves room for the one floating-point difference that
is not an identity, ``1 - (1 - x)`` against ``x``, and for nothing else.

Designs are drawn so that no level separates (every level keeps a success and a failure on both
sides) and so that the rates trend across the design. Estimated-asymptote cases use five or six
levels, because four free parameters cannot be identified from fewer. Whatever status results is
compared too, so a fit that fails to converge is checked for failing identically.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from robustmargin import (
    Axis,
    Definition,
    Fit,
    Observations,
    Status,
    Threshold,
    check_monotonicity,
    fit_dose_response,
    threshold,
)

_LEVEL = 0.95
_REL = 1e-9
_ABS = 1e-12
_LINKS = ("probit", "logit", "cloglog")
_AXES = {
    "log": Axis(name="severity", unit="unit", scale="log"),
    "linear": Axis(name="severity", unit="unit", scale="linear"),
}

Asymptote = float | str

# (upper, lower) on the *outcome* scale of the increasing fit.
_ASYMPTOTE_CASES: dict[str, tuple[Asymptote, Asymptote]] = {
    "fixed_1_0": (1.0, 0.0),
    "estimate_upper": ("estimate", 0.0),
    "estimate_lower": (1.0, "estimate"),
    "estimate_both": ("estimate", "estimate"),
    "fixed_inner": (0.95, 0.05),
}
_SWAP = {"mu": "mu", "s": "s", "upper": "lower", "lower": "upper"}
_ASYMPTOTE_NAMES = frozenset({"upper", "lower"})

Counts = tuple[list[float], list[int], list[int]]


def _mirror_one(value: Asymptote) -> Asymptote:
    return value if isinstance(value, str) else 1.0 - value


def _mirrored_asymptotes(upper: Asymptote, lower: Asymptote) -> tuple[Asymptote, Asymptote]:
    """``(u', l') = (1 - l, 1 - u)``, with ``"estimate"`` unchanged."""
    return _mirror_one(lower), _mirror_one(upper)


@st.composite
def _increasing_counts(draw: st.DrawFn, *, min_levels: int, max_levels: int) -> Counts:
    """Events out of trials, rising with severity across most of the unit range, with every level
    carrying at least one event and one non-event (so nothing separates)."""
    n = draw(st.integers(min_value=min_levels, max_value=max_levels))
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
    bottom = draw(st.floats(min_value=0.05, max_value=0.20))
    top = draw(st.floats(min_value=0.80, max_value=0.95))
    step = (top - bottom) / (n - 1)
    events = [
        min(max(int(round((bottom + i * step) * n_trials)), 1), n_trials - 1)
        for i, n_trials in enumerate(trials)
    ]
    return severities, events, trials


_FEW_LEVELS = _increasing_counts(min_levels=3, max_levels=4)
_MANY_LEVELS = _increasing_counts(min_levels=5, max_levels=6)


def _observations(counts: Counts, *, axis: Axis, direction: str, complement: bool) -> Observations:
    severity, events, trials = counts
    successes = [n - k for k, n in zip(events, trials, strict=True)] if complement else events
    return Observations.from_counts(
        axis,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="outcome",
        direction=direction,
    )


def _fit_pair(
    counts: Counts, *, scale: str, link: str, upper: Asymptote, lower: Asymptote
) -> tuple[Fit, Fit]:
    """The increasing fit, and the decreasing fit on the complement with mirrored asymptotes."""
    axis = _AXES[scale]
    upper_dec, lower_dec = _mirrored_asymptotes(upper, lower)
    increasing = fit_dose_response(
        _observations(counts, axis=axis, direction="increasing", complement=False),
        model="binomial",
        link=link,
        upper=upper,
        lower=lower,
    )
    decreasing = fit_dose_response(
        _observations(counts, axis=axis, direction="decreasing", complement=True),
        model="binomial",
        link=link,
        upper=upper_dec,
        lower=lower_dec,
    )
    return increasing, decreasing


def _cov_entry(fit: Fit, a: str, b: str) -> float:
    assert fit.covariance is not None
    names = fit.covariance.names
    return fit.covariance.matrix[names.index(a)][names.index(b)]


def _assert_fits_mirror(increasing: Fit, decreasing: Fit, *, counts: Counts, axis: Axis) -> None:
    assert increasing.direction == "increasing"
    assert decreasing.direction == "decreasing"
    assert increasing.status is decreasing.status

    # Carried from the original Observations, never complemented.
    severity, events, trials = counts
    assert [(c.severity, c.successes, c.trials) for c in increasing.cells] == list(
        zip(severity, events, trials, strict=True)
    )
    original = _observations(counts, axis=axis, direction="increasing", complement=False)
    assert increasing.monotonicity == check_monotonicity(original)

    if increasing.status is not Status.OK:
        assert increasing.params is None and increasing.covariance is None
        assert increasing.log_likelihood is None
        return

    assert increasing.params is not None and decreasing.params is not None
    assert increasing.covariance is not None and decreasing.covariance is not None
    assert increasing.log_likelihood == pytest.approx(decreasing.log_likelihood, rel=_REL, abs=_ABS)
    for name in ("mu", "s"):
        assert increasing.params[name].value == pytest.approx(
            decreasing.params[name].value, rel=_REL, abs=_ABS
        )
        assert increasing.params[name].fixed is decreasing.params[name].fixed
    for name in ("upper", "lower"):
        mirrored = decreasing.params[_SWAP[name]]
        assert increasing.params[name].value == pytest.approx(
            1.0 - mirrored.value, rel=_REL, abs=_ABS
        )
        assert increasing.params[name].fixed is mirrored.fixed

    assert set(increasing.covariance.names) == {_SWAP[n] for n in decreasing.covariance.names}
    for a, b in itertools.product(increasing.covariance.names, repeat=2):
        sign = (-1.0 if a in _ASYMPTOTE_NAMES else 1.0) * (-1.0 if b in _ASYMPTOTE_NAMES else 1.0)
        assert _cov_entry(increasing, a, b) == pytest.approx(
            sign * _cov_entry(decreasing, _SWAP[a], _SWAP[b]), rel=_REL, abs=1e-14
        )

    # Diagnostics are those of the mirrored fit.
    assert increasing.diagnostics is not None and decreasing.diagnostics is not None
    assert increasing.diagnostics.deviance == pytest.approx(
        decreasing.diagnostics.deviance, rel=_REL, abs=_ABS
    )
    assert increasing.diagnostics.dispersion == pytest.approx(
        decreasing.diagnostics.dispersion, rel=_REL, abs=_ABS
    )
    assert increasing.diagnostics.df == decreasing.diagnostics.df


@pytest.mark.parametrize("link", _LINKS)
@pytest.mark.parametrize("case", ["fixed_1_0", "fixed_inner"])
@given(counts=_FEW_LEVELS, scale=st.sampled_from(["log", "linear"]))
@settings(max_examples=15, deadline=None)
def test_increasing_fit_equals_the_decreasing_fit_on_the_complement_with_fixed_asymptotes(
    link: str, case: str, counts: Counts, scale: str
) -> None:
    upper, lower = _ASYMPTOTE_CASES[case]

    increasing, decreasing = _fit_pair(counts, scale=scale, link=link, upper=upper, lower=lower)

    _assert_fits_mirror(increasing, decreasing, counts=counts, axis=_AXES[scale])


@pytest.mark.parametrize("link", _LINKS)
@pytest.mark.parametrize("case", ["estimate_upper", "estimate_lower", "estimate_both"])
@given(counts=_MANY_LEVELS, scale=st.sampled_from(["log", "linear"]))
@settings(max_examples=8, deadline=None)
def test_increasing_fit_equals_the_decreasing_fit_on_the_complement_with_estimated_asymptotes(
    link: str, case: str, counts: Counts, scale: str
) -> None:
    upper, lower = _ASYMPTOTE_CASES[case]

    increasing, decreasing = _fit_pair(counts, scale=scale, link=link, upper=upper, lower=lower)

    _assert_fits_mirror(increasing, decreasing, counts=counts, axis=_AXES[scale])


def _assert_predict_bands_mirror(
    case: str, counts: Counts, scale: str, link: str, *, require_ok: bool = False
) -> None:
    upper, lower = _ASYMPTOTE_CASES[case]
    increasing, decreasing = _fit_pair(counts, scale=scale, link=link, upper=upper, lower=lower)
    if require_ok:
        assert increasing.status is Status.OK
    if increasing.status is not Status.OK:
        assert decreasing.status is increasing.status
        return
    severities = sorted({*counts[0], 0.75 * counts[0][0], 1.5 * counts[0][-1]})

    inc = increasing.predict(severities, level=_LEVEL)
    dec = decreasing.predict(severities, level=_LEVEL)

    # The band can be nan in an extrapolated tail where the curve rounds onto a fixed 0/1
    # asymptote. That is a known defect of the band itself; the mirror property still holds there,
    # so a nan counts as mirrored only when it sits at the same index in both orientations (a nan
    # on one side only still fails). The point estimate is never nan.
    assert np.isfinite(inc.estimate).all()
    assert np.isfinite(dec.estimate).all()
    assert inc.estimate == pytest.approx(1.0 - dec.estimate, rel=_REL, abs=_ABS)
    assert inc.lo == pytest.approx(1.0 - dec.hi, rel=_REL, abs=_ABS, nan_ok=True)
    assert inc.hi == pytest.approx(1.0 - dec.lo, rel=_REL, abs=_ABS, nan_ok=True)
    assert np.array_equal(np.isnan(inc.lo), np.isnan(dec.hi))
    assert np.array_equal(np.isnan(inc.hi), np.isnan(dec.lo))
    lo_finite = np.isfinite(inc.lo)
    hi_finite = np.isfinite(inc.hi)
    assert (inc.lo[lo_finite] <= inc.estimate[lo_finite]).all()
    assert (inc.estimate[hi_finite] <= inc.hi[hi_finite]).all()


@pytest.mark.parametrize("case", ["fixed_1_0", "estimate_lower", "estimate_both"])
@given(counts=_MANY_LEVELS, scale=st.sampled_from(["log", "linear"]), link=st.sampled_from(_LINKS))
@settings(max_examples=8, deadline=None)
def test_predict_on_an_increasing_fit_is_the_mirrored_band_of_the_decreasing_fit(
    case: str, counts: Counts, scale: str, link: str
) -> None:
    _assert_predict_bands_mirror(case, counts, scale, link)


def test_predict_band_that_is_nan_in_an_extrapolated_tail_is_nan_in_both_orientations() -> None:
    """Pinned case: an OK estimated-lower cloglog fit whose band is nan at 1.5 x the largest
    severity, in both orientations. (A parametrized ``case`` cannot be supplied through
    ``@example``, so the pinned case calls the shared check directly.)"""
    counts: Counts = ([1.9, 2.0, 4.32, 12.85, 13.57, 15.32], [12, 25, 38, 50, 63, 76], [80] * 6)

    _assert_predict_bands_mirror("estimate_lower", counts, "log", "cloglog", require_ok=True)


def _assert_thresholds_mirror(inc: Threshold, dec: Threshold) -> None:
    assert inc.direction == "increasing"
    assert dec.direction == "decreasing"
    assert inc.status is dec.status
    assert inc.censoring is dec.censoring
    assert inc.interval_method == dec.interval_method
    assert inc.grid == dec.grid
    for field in ("value", "lo", "hi"):
        a, b = getattr(inc, field), getattr(dec, field)
        if b is None:
            assert a is None, f"{field}: increasing gave {a!r}, mirrored decreasing gave None"
        else:
            assert a == pytest.approx(b, rel=_REL, abs=_ABS), field


@pytest.mark.parametrize("interval_method", ["profile", "delta"])
@pytest.mark.parametrize("case", ["fixed_1_0", "estimate_upper", "estimate_lower", "fixed_inner"])
@given(counts=_MANY_LEVELS, scale=st.sampled_from(["log", "linear"]), link=st.sampled_from(_LINKS))
@settings(max_examples=6, deadline=None)
def test_increasing_threshold_equals_the_mirrored_decreasing_threshold(
    interval_method: str, case: str, counts: Counts, scale: str, link: str
) -> None:
    upper, lower = _ASYMPTOTE_CASES[case]
    increasing, decreasing = _fit_pair(counts, scale=scale, link=link, upper=upper, lower=lower)
    assert increasing.status is decreasing.status

    pairs: list[tuple[Definition, Definition]] = [
        (Definition.absolute(a), Definition.absolute(1.0 - a)) for a in (0.3, 0.5, 0.7)
    ]
    if increasing.status is Status.OK:
        # relative(p) needs fitted asymptotes, so it is only defined on an OK fit.
        pairs += [(Definition.relative(p), Definition.relative(p)) for p in (0.25, 0.5)]

    for definition_inc, definition_dec in pairs:
        t_inc = threshold(
            increasing, definition=definition_inc, interval_method=interval_method, level=_LEVEL
        )
        t_dec = threshold(
            decreasing, definition=definition_dec, interval_method=interval_method, level=_LEVEL
        )

        _assert_thresholds_mirror(t_inc, t_dec)
        assert t_inc.fit is increasing, "Threshold.fit is the original, unmirrored Fit"
        assert t_inc.definition == definition_inc


@pytest.mark.parametrize("interval_method", ["profile", "delta"])
@given(counts=_FEW_LEVELS, link=st.sampled_from(_LINKS))
@settings(max_examples=12, deadline=None)
def test_increasing_threshold_is_mirrored_on_exact_failed_fits_too(
    interval_method: str, counts: Counts, link: str
) -> None:
    """Separated data: the exact-bound fallback mirrors as well. Events are zeroed below some
    level and saturated at and above it, which separates completely."""
    severity, _events, trials = counts
    cut = len(severity) // 2
    separated = (
        severity,
        [0] * cut + [n for n in trials[cut:]],
        trials,
    )
    increasing, decreasing = _fit_pair(separated, scale="log", link=link, upper=1.0, lower=0.0)
    assert increasing.status is Status.SEPARATION
    assert decreasing.status is Status.SEPARATION

    for a in (0.2, 0.5, 0.8):
        t_inc = threshold(
            increasing,
            definition=Definition.absolute(a),
            interval_method=interval_method,
            level=_LEVEL,
        )
        t_dec = threshold(
            decreasing,
            definition=Definition.absolute(1.0 - a),
            interval_method=interval_method,
            level=_LEVEL,
        )

        _assert_thresholds_mirror(t_inc, t_dec)
