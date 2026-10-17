"""Real-data regression: ``threshold()`` on ``tests/data/pooled_counts.csv``.

Every series here is a recorded edge case, not a synthetic construction -- a status or censoring
change on this file is either a real statistical finding or a real bug, never something to paper
over by adjusting the assertion. Status and censoring are asserted for every series; numeric
``value``/``lo``/``hi`` are asserted only for the identifiable ``system_a_axis_1`` series, because
the other series are separated, censored or non-monotone and have no interior estimate to pin.

Status/Censoring member coverage *not* reachable from this CSV (``CONTROL_INCOMPATIBLE``,
``NOT_CONVERGED``, ``OPEN_UPPER``/``OPEN_LOWER``, and the ``UNREACHABLE`` routes) lives in
``tests/unit/test_threshold_status_coverage.py`` instead, on constructed data -- kept separate
so this file stays a pure snapshot of the real matrix.
"""

from __future__ import annotations

import csv
import dataclasses
import re
from pathlib import Path

import pytest

from robustmargin import (
    Axis,
    Censoring,
    Definition,
    Observations,
    Status,
    fit_dose_response,
    threshold,
)
from robustmargin.censoring import classify
from robustmargin.testing import fake_fit

_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "pooled_counts.csv"
_LEVEL = 0.95


def _load_series() -> dict[str, tuple[list[float], list[int], list[int]]]:
    series: dict[str, tuple[list[float], list[int], list[int]]] = {}
    with _CSV_PATH.open(newline="") as fh:
        for row in csv.DictReader(fh):
            name = row["series"]
            severity, successes, trials = series.setdefault(name, ([], [], []))
            severity.append(float(row["severity"]))
            successes.append(int(row["successes"]))
            trials.append(int(row["trials"]))
    return series


_SERIES = _load_series()


def _observations_for(
    series_name: str, *, axis_name: str, axis_unit: str, axis_scale: str
) -> Observations:
    severity, successes, trials = _SERIES[series_name]
    axis = Axis(name=axis_name, unit=axis_unit, scale=axis_scale)
    return Observations.from_counts(
        axis,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )


class TestSystemAAxis1IdentifiableTransition:
    """The one series with numeric assertions: 0: 200/200, 0.01: 299/300, 0.05: 122/300,
    0.1: 0/300 -- a clean, monotone, identifiable transition.
    """

    def test_absolute_0_95_gives_ok_status_none_censoring_and_a_finite_bracketed_interval(
        self,
    ) -> None:
        obs = _observations_for(
            "system_a_axis_1", axis_name="axis_1", axis_unit="m", axis_scale="log"
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
        assert fit.status is Status.OK

        t = threshold(
            fit, definition=Definition.absolute(0.95), interval_method="profile", level=_LEVEL
        )

        assert t.status is Status.OK
        assert t.censoring is Censoring.NONE
        assert t.interval_method in ("profile", "delta")
        assert t.value is not None and t.lo is not None and t.hi is not None
        assert t.lo <= t.value <= t.hi
        assert 0.01 < t.value < 0.1
        assert t.baseline is None
        assert t.grid is None


class TestSystemAAxis2RightCensoredCompleteSeparation:
    """500/500 at every tested wind level (2, 5, 10 m/s), 100/100 at the zero control: complete
    separation (no failure anywhere), so ``fit_dose_response`` itself reports
    ``Status.SEPARATION`` (confirmed directly: ``tests/unit/test_models_pooled_counts.py::
    test_system_a_axis_2_u_1_l_0_is_separation``). The exact one-sided 95% lower bound on the
    success rate at 10 m/s is ``0.05**(1/500) ~= 0.994``, well above 0.95, so the result is
    ``RIGHT`` with the bound reported at ``lo=10.0`` -- independent of the fit having converged to
    an interior estimate at all.
    """

    def test_absolute_0_95_is_right_censored_at_10_with_grid_attached(self) -> None:
        obs = _observations_for(
            "system_a_axis_2", axis_name="axis_2", axis_unit="m/s", axis_scale="linear"
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
        assert fit.status is Status.SEPARATION

        t = threshold(
            fit, definition=Definition.absolute(0.95), interval_method="profile", level=_LEVEL
        )

        assert t.censoring is Censoring.RIGHT
        assert t.value is None
        assert t.lo == 10.0
        assert t.hi is None
        assert t.interval_method == "exact_bound"
        # RIGHT/LEFT take the fit's status -- the fit never reached an interior estimate, and the
        # threshold says so by carrying the same status forward.
        assert t.status is fit.status is Status.SEPARATION
        # `grid` is set on every SEPARATION/NOT_CONVERGED exact-bound fallback.
        assert t.grid is not None
        assert t.grid.max_tested == 10.0
        assert t.baseline is None


class TestSystemBAxis1FailsAtBaseline:
    """The control is already at 98/200 (rate 0.49); every nonzero level is 0/300, for
    ``absolute(0.95)``. The underlying fit is ``Status.SEPARATION`` here
    (confirmed: ``tests/unit/test_models_pooled_counts.py::test_system_b_axis_1_upper_estimate_is_
    separation`` -- no success at all is observed at any nonzero level, so
    ``fit_dose_response``'s pre-fit separation check fires unconditionally, for *any* ``upper``
    argument), but ``FAILS_AT_BASELINE`` places no constraint on ``fit.status``: it is decided
    from an exact one-sided test of the raw control cell alone, before -- and regardless of --
    whatever the rest of the curve does.
    """

    def test_absolute_0_95_fails_at_baseline_with_the_control_rate_attached(self) -> None:
        obs = _observations_for(
            "system_b_axis_1", axis_name="axis_1", axis_unit="m", axis_scale="log"
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)

        t = threshold(
            fit, definition=Definition.absolute(0.95), interval_method="profile", level=_LEVEL
        )

        assert t.status is Status.FAILS_AT_BASELINE
        assert t.censoring is Censoring.NONE
        assert t.value is None and t.lo is None and t.hi is None
        assert t.interval_method == "exact_bound"

        # The control rate travels on Threshold.baseline, with its own exact CI, never smuggled
        # into .lo/.hi: a per-cell exact rate is not an interval for the threshold itself.
        assert t.baseline is not None
        assert t.baseline.severity == 0.0
        assert t.baseline.successes == 98
        assert t.baseline.trials == 200
        assert t.baseline.rate == pytest.approx(0.49)
        assert t.baseline.method == "per_cell_clopper_pearson"
        assert t.baseline.lo <= t.baseline.rate <= t.baseline.hi


class TestSystemBAxis1LeftCensored:
    """``baseline_fraction(0.5)`` gives a target ``P* ~= 0.245`` (half the control's own 98/200
    rate: the fallback path has no fitted ``u``, so the control's *observed* rate is used). At
    severity 0.01 the result is
    0/300, whose exact one-sided 95% upper bound is ``1 - 0.05**(1/300) ~= 0.0099`` -- far below
    0.245 -- while the control (98/200) clears 0.245 easily, so the result is ``LEFT`` with the
    bound reported at ``hi=0.01``.
    """

    def test_baseline_fraction_0_5_is_left_censored_at_0_01(self) -> None:
        obs = _observations_for(
            "system_b_axis_1", axis_name="axis_1", axis_unit="m", axis_scale="log"
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)
        assert fit.status is Status.SEPARATION

        t = threshold(
            fit,
            definition=Definition.baseline_fraction(0.5),
            interval_method="profile",
            level=_LEVEL,
        )

        assert t.censoring is Censoring.LEFT
        assert t.value is None
        assert t.hi == 0.01
        assert t.lo is None
        assert t.interval_method == "exact_bound"
        assert t.status is fit.status is Status.SEPARATION
        assert t.baseline is None
        assert t.grid is not None


class TestSystemCAxis1NonMonotoneContradiction:
    """The rate *rises* from severity 0 (57/200) to 0.01 (143/300) before collapsing to 0/300 at
    0.05 and 0.1 -- the non-monotone edge case. The fit is ``Status.SEPARATION``
    (confirmed: ``tests/unit/test_models_pooled_counts.py::
    test_system_c_axis_1_upper_estimate_is_separation_quasi_complete_at_0_01``), which is what
    makes the contradiction encoding reachable at all -- ``Threshold`` only ever reports a
    contradiction on the ``SEPARATION``/``NOT_CONVERGED`` fallback path.

    At ``level=0.95``, the exact one-sided bounds on the two lowest severities are
    ``x=0.0: [0.2327, 0.3422]`` and ``x=0.01: [0.4279, 0.5258]`` (each individually one-sided at
    ``level``, matching ``exact_one_sided_bound``'s own convention -- ``tests/unit/
    test_censoring.py``). For any target in roughly ``(0.3422, 0.4279)``, x=0.0 (the *lower*
    severity) is confidently classified as failing (its upper bound sits below the target) while
    x=0.01 (the *higher* severity) is confidently classified as passing (its lower bound sits
    above the target) -- backwards from what a monotone decreasing curve allows, so the ordinary
    bracket construction would need ``lo=0.01 > hi=0.0``, which ``Threshold.__post_init__``
    already refuses to build. The rule is to report no bounds at all, name both levels in
    ``warnings``, and keep ``status``/``grid``.
    """

    _TARGET = 0.4  # inside (0.3422, 0.4279): see the exact bounds derivation above.

    def test_absolute_0_4_reports_no_bounds_with_both_contradicting_levels_named(self) -> None:
        obs = _observations_for(
            "system_c_axis_1", axis_name="axis_1", axis_unit="m", axis_scale="log"
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)
        assert fit.status is Status.SEPARATION

        t = threshold(
            fit,
            definition=Definition.absolute(self._TARGET),
            interval_method="profile",
            level=_LEVEL,
        )

        assert t.status is fit.status is Status.SEPARATION
        assert t.censoring is Censoring.NONE
        assert t.value is None
        assert t.lo is None
        assert t.hi is None
        assert t.interval_method == "exact_bound"

        # The contradiction is named in warnings, not silently resolved by keeping one bound.
        # "0.01" is a severity value unique to the higher of the two contradicting levels; the
        # lower (0.0) is matched with a regex that requires it not be immediately followed by a
        # "1" digit, so a warning that only mentions "0.01" cannot accidentally satisfy the "0.0"
        # check too (Python's own `"0.0" in "0.01"` is True, which would make that check
        # vacuous).
        combined = " ".join(t.warnings)
        assert combined, "a contradiction must be named in a warning"
        assert "0.01" in combined
        assert re.search(r"0\.0(?!1)", combined) is not None

        # The grid fallback is still attached, exactly as on any other SEPARATION/NOT_CONVERGED
        # fallback -- it makes no monotonicity assumption, so it stays meaningful precisely where
        # the exact bounds have stopped being so.
        assert t.grid is not None

    def test_warning_starts_with_the_non_monotone_token(self) -> None:
        """The ``NON_MONOTONE: `` token is the supported way a consumer detects this condition. A
        *separate*, adjacent-pair monotonicity check on ``Fit`` (``diagnose()``) leaves this
        token unchanged (changing that token is breaking) -- pinned here directly, since
        ``censoring.py`` relies on it as load-bearing.
        """
        obs = _observations_for(
            "system_c_axis_1", axis_name="axis_1", axis_unit="m", axis_scale="log"
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)
        assert fit.status is Status.SEPARATION

        t = threshold(
            fit,
            definition=Definition.absolute(self._TARGET),
            interval_method="profile",
            level=_LEVEL,
        )

        assert any(w.startswith("NON_MONOTONE: ") for w in t.warnings)

    def test_warning_does_not_mention_a_fitted_curve_when_lo_and_hi_are_both_none(self) -> None:
        """`censoring._contradiction_warning` is shared by the converged path
        (``_contradiction_or_none``, where ``lo``/``hi`` genuinely do come from the fitted
        curve's own profile/delta interval) and this failed-fit fallback
        (``_exact_bound_fallback``/``_contradiction_or_none``'s non-OK branch, where there is no
        fitted curve at all and ``t.lo``/``t.hi`` are both ``None``). The wording is only honest
        on the first of those two callers -- on this one, a phrase like "any lo/hi this
        Threshold carries comes from the fitted curve instead" implies bounds exist somewhere on
        this object, which is false: there is no fitted curve (``status`` is ``SEPARATION``) and
        no bounds at all.
        """
        obs = _observations_for(
            "system_c_axis_1", axis_name="axis_1", axis_unit="m", axis_scale="log"
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)
        assert fit.status is Status.SEPARATION

        t = threshold(
            fit,
            definition=Definition.absolute(self._TARGET),
            interval_method="profile",
            level=_LEVEL,
        )

        assert t.status is Status.SEPARATION
        assert t.lo is None
        assert t.hi is None

        non_monotone = next(w for w in t.warnings if w.startswith("NON_MONOTONE: "))
        assert "fitted curve" not in non_monotone, (
            "misleading on the failed-fit fallback -- there is no fitted curve, and this "
            f"Threshold's lo/hi are both None: {non_monotone!r}"
        )
        assert "fitted model" not in non_monotone, (
            f"same defect, different wording: {non_monotone!r}"
        )


class TestSystemCAxis3SignedAxis:
    """``system_c_axis_3`` is signed (``-0.2`` .. ``+0.2``); ``Observations`` rejects negative
    severity outright (a signed axis is handled by the caller splitting it into two
    magnitude-based ``Observations``, one per side -- the schema does not model a signed scalar
    axis at all). **What this skips:** there is no way to call the public API with the raw
    signed ``system_c_axis_3`` series directly (``Observations.from_counts`` raises on the negative
    ``-0.2``/``-0.1`` rows), so this only covers the magnitude-split form, matching
    ``tests/unit/test_models_pooled_counts.py::_system_c_axis_3_by_magnitude``'s own split.

    Both magnitude series converge to ``Status.OK`` with fixed ``upper=1.0, lower=0.0``
    (confirmed: ``test_system_c_axis_3_positive_magnitude_u_1_l_0_is_ok`` /
    ``..._negative_magnitude_..._is_ok``), and each fitted ``mu`` (~0.084 positive side, ~0.13
    negative side) lies strictly inside its own tested range ``[0.0, 0.2]``. **This does not
    give ``NONE`` censoring with a profile interval** -- the ``RIGHT``/``LEFT`` conditions are
    evaluated on the data and outrank a converged fit.
    On both magnitude sides, the control passes ``absolute(0.95)`` (exact one-sided lower bound
    ``0.9534 > 0.95`` on the shared ``99/100`` control) while **no nonzero level passes**:
    positive ``0.1: 33/100`` has exact upper bound ``0.4155``; negative ``0.1: 77/100`` has exact
    upper bound ``0.8374``; both well below ``0.95``, and ``0.2`` fails even more clearly on
    either side. This differs from ``system_a_axis_1`` (the plain interior case), where
    ``0.01 -> 299/300`` genuinely passes and ``0.05 -> 122/300`` fails --
    *some* nonzero level passes there, so the crossing interpolates between two measured points
    and keeps its number. Here, with no nonzero level passing on either side, the crossing is
    only bracketed by the unmeasured control-to-``0.1`` gap, so it is ``LEFT`` with the bound at
    the lowest nonzero severity (``0.1``), on both sides, despite the fit itself being ``OK``
    with an interior ``mu``.

    ``(Status.OK, Censoring.LEFT)`` is one of the fourteen ``(status, censoring)`` pairs that
    ``robustmargin.testing.fake_threshold``'s ``_ALLOWED_THRESHOLD_COMBINATIONS`` allows -- this
    is what makes it reachable at all, not a new invariant.
    """

    @staticmethod
    def _magnitude_split() -> tuple[Observations, Observations]:
        severity, successes, trials = _SERIES["system_c_axis_3"]
        by_severity = dict(zip(severity, zip(successes, trials, strict=True), strict=True))
        control_successes, control_trials = by_severity[0.0]

        positive_axis = Axis(name="axis_3_positive", unit="kg", scale="linear")
        positive = Observations.from_counts(
            positive_axis,
            severity=[0.0, 0.1, 0.2],
            successes=[control_successes, *(by_severity[s][0] for s in (0.1, 0.2))],
            trials=[control_trials, *(by_severity[s][1] for s in (0.1, 0.2))],
            outcome="success",
            direction="decreasing",
        )
        negative_axis = Axis(name="axis_3_negative", unit="kg", scale="linear")
        negative = Observations.from_counts(
            negative_axis,
            severity=[0.0, 0.1, 0.2],
            successes=[control_successes, *(by_severity[-s][0] for s in (0.1, 0.2))],
            trials=[control_trials, *(by_severity[-s][1] for s in (0.1, 0.2))],
            outcome="success",
            direction="decreasing",
        )
        return positive, negative

    def test_negative_severity_is_rejected_by_the_public_api(self) -> None:
        axis = Axis(name="axis_3", unit="kg", scale="linear")
        with pytest.raises(ValueError):
            Observations.from_counts(
                axis,
                severity=[-0.2, -0.1, 0.0, 0.1, 0.2],
                successes=[5, 77, 99, 33, 0],
                trials=[100, 100, 100, 100, 100],
                outcome="success",
                direction="decreasing",
            )

    def test_positive_magnitude_split_absolute_0_95_is_left_censored_at_0_1(self) -> None:
        """The control passes but no nonzero level does, so the converged ``OK`` fit's own
        interior ``mu`` is not reported -- see the class docstring for the exact-bound
        arithmetic.
        """
        positive, _negative = self._magnitude_split()
        fit = fit_dose_response(positive, model="binomial", link="probit", upper=1.0, lower=0.0)
        assert fit.status is Status.OK

        t = threshold(
            fit, definition=Definition.absolute(0.95), interval_method="profile", level=_LEVEL
        )

        assert t.status is Status.OK
        assert t.censoring is Censoring.LEFT
        assert t.value is None
        assert t.lo is None
        assert t.hi == 0.1
        assert t.interval_method == "exact_bound"

    def test_negative_magnitude_split_absolute_0_95_is_left_censored_at_0_1(self) -> None:
        """Mirrors the positive side -- the control passes but no nonzero level
        does (``0.1: 77/100``'s exact upper bound is ``0.8374``, still below ``0.95``), so this
        is also ``LEFT`` despite the fit converging to an interior ``mu`` (~0.13).
        """
        _positive, negative = self._magnitude_split()
        fit = fit_dose_response(negative, model="binomial", link="probit", upper=1.0, lower=0.0)
        assert fit.status is Status.OK

        t = threshold(
            fit, definition=Definition.absolute(0.95), interval_method="profile", level=_LEVEL
        )

        assert t.status is Status.OK
        assert t.censoring is Censoring.LEFT
        assert t.value is None
        assert t.lo is None
        assert t.hi == 0.1
        assert t.interval_method == "exact_bound"


class TestRealSeriesReachingTheExactBoundFallbackAreUnchangedByFixedSequenceScans:
    """The failed-fit exact bracket is built by fixed-sequence scans from each end of the
    curve, not by a max-anywhere rule. Where the confidently passing and failing levels are each
    contiguous from their end, the two rules agree, and every real series here is such data --
    so the bounds below are the same under both, and a change to any of them is a behavioural
    change to the exact bracket, not noise.

    Only series whose fit is a *deterministic* failed fit are run through ``fit_dose_response``:
    ``system_b_axis_1`` and ``system_c_axis_1`` at ``upper="estimate"`` are ``SEPARATION`` by
    the pre-fit separation check (asserted in ``test_models_pooled_counts.py``). ``system_a_axis_1``
    converges at ``upper=1.0`` and at ``upper="estimate"`` its status is optimiser-dependent, so
    its real cells are classified against a constructed failed fit instead.
    """

    def test_system_c_axis_1_absolute_0_3_is_a_bracket_between_0_01_and_0_05(self) -> None:
        """The non-monotone-looking series (57/200 at the control, 143/300 at 0.01, then 0/300
        twice). At a target of 0.3 the control is undetermined (exact bounds 0.2327 / 0.3422),
        0.01 confidently passes (lower bound above 0.3) and 0.05 and 0.1 confidently fail: a
        clean bracket, with no ``NON_MONOTONE: `` contradiction (contrast the 0.4 target in
        ``TestSystemCAxis1NonMonotoneContradiction``, which does contradict)."""
        obs = _observations_for(
            "system_c_axis_1", axis_name="axis_1", axis_unit="m", axis_scale="log"
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)
        assert fit.status is Status.SEPARATION

        t = threshold(
            fit, definition=Definition.absolute(0.3), interval_method="profile", level=_LEVEL
        )

        assert t.status is Status.SEPARATION
        assert t.censoring is Censoring.NONE
        assert t.value is None
        assert t.lo == 0.01
        assert t.hi == 0.05
        assert t.interval_method == "exact_bound"
        assert t.grid is not None
        assert not any(w.startswith("NON_MONOTONE: ") for w in t.warnings)

    def test_system_b_axis_1_absolute_0_5_is_a_one_sided_upper_bound_at_0_01(self) -> None:
        """Control 98/200 is undetermined against 0.5 (exact bounds 0.4298 / 0.5505) so it ends
        the run from the top, and 0/300 at 0.01, 0.05 and 0.1 all confidently fail: the run is
        0.1, 0.05, 0.01. No nonzero level passes, so only the upper side is determined."""
        obs = _observations_for(
            "system_b_axis_1", axis_name="axis_1", axis_unit="m", axis_scale="log"
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)
        assert fit.status is Status.SEPARATION

        t = threshold(
            fit, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL
        )

        assert t.status is Status.SEPARATION
        assert t.censoring is Censoring.NONE
        assert t.value is None
        assert t.lo is None
        assert t.hi == 0.01
        assert t.interval_method == "exact_bound"
        assert t.grid is not None
        assert any("determined only the upper side" in w for w in t.warnings)

    @pytest.mark.parametrize(
        ("target", "lo", "hi"),
        [(0.95, 0.01, 0.05), (0.5, 0.01, 0.05), (0.3, 0.05, 0.1)],
        ids=["absolute_0.95", "absolute_0.5", "absolute_0.3"],
    )
    @pytest.mark.parametrize("status", [Status.SEPARATION, Status.NOT_CONVERGED])
    def test_system_a_axis_1_cells_on_a_failed_fit_give_the_same_bracket(
        self, status: Status, target: float, lo: float, hi: float
    ) -> None:
        """The cleanest real series (200/200, 299/300, 122/300, 0/300): passes then fails with
        no undetermined level in between at 0.95 and 0.5, and with 0.01 and 0.05 both passing
        and only 0.1 failing at 0.3. Classified against a failed fit carrying exactly these
        cells (``classify`` reads only ``status`` and ``cells``)."""
        obs = _observations_for(
            "system_a_axis_1", axis_name="axis_1", axis_unit="m", axis_scale="log"
        )
        failed = dataclasses.replace(fake_fit(status=status, monotonicity=None), cells=obs.cells())

        result = classify(failed, target=target, level=_LEVEL)

        assert result.status is status
        assert result.censoring is Censoring.NONE
        assert result.lo == lo
        assert result.hi == hi
        assert result.warnings == ()
        assert result.grid is not None
