"""``Status``/``Censoring`` member coverage for ``threshold()`` that the pooled-counts fixture
(``tests/unit/test_threshold_pooled_counts.py``) does not reach on its own:
``CONTROL_INCOMPATIBLE``, ``NOT_CONVERGED``, both ``UNREACHABLE`` routes, the ordering rule
between ``FAILS_AT_BASELINE`` and ``UNREACHABLE`` when both apply to the same fit, and
``OPEN_UPPER``/``OPEN_LOWER``.

Every dataset here is constructed, not drawn from ``pooled_counts.csv`` -- each is chosen
because the *real* ``fit_dose_response`` (confirmed empirically, not assumed) produces the fit
status the corresponding ``threshold()`` case needs, so this file is still checking real fit
behaviour, only not real-world data.
"""

from __future__ import annotations

import json
from pathlib import Path

from robustmargin import Axis, Definition, Fit, Observations, Status, fit_dose_response, threshold
from robustmargin.censoring import exact_one_sided_bound
from robustmargin.types import Censoring

_LEVEL = 0.95
_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tools" / "drc_reference"


class TestControlIncompatible:
    """A fixed ``upper=1.0`` with a failing zero-severity control drives the likelihood to zero on a
    log axis; ``fit_dose_response`` reports ``Status.CONTROL_INCOMPATIBLE`` before ever attempting
    to fit (confirmed:
    ``tests/unit/test_models_pooled_counts.py::test_system_b_axis_1_u_1_is_control_incompatible``,
    on the same ``system_b_axis_1`` counts reused here).
    """

    def test_threshold_mirrors_control_incompatible_with_no_numbers(self) -> None:
        axis = Axis(name="axis_1", unit="m", scale="log")
        obs = Observations.from_counts(
            axis,
            severity=[0.0, 0.01, 0.05, 0.1],
            successes=[98, 0, 0, 0],
            trials=[200, 300, 300, 300],
            outcome="success",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
        assert fit.status is Status.CONTROL_INCOMPATIBLE

        t = threshold(
            fit, definition=Definition.absolute(0.95), interval_method="profile", level=_LEVEL
        )

        assert t.status is fit.status is Status.CONTROL_INCOMPATIBLE
        assert t.censoring is Censoring.NONE
        assert t.value is None and t.lo is None and t.hi is None
        assert t.interval_method == "exact_bound"
        assert t.baseline is None


class TestNotConverged:
    """A ``NOT_CONVERGED`` fit reached through the **wrong-sign** check, which
    is a sign test on the fitted slope rather than an optimiser tolerance.

    Success *rises* with severity here while ``direction="decreasing"``, so the constrained
    maximum (``s > 0``) is the flat curve and no finite ``(mu, s)`` exists. That verdict is
    robustmargin's own post-fit check on a slope that is unambiguously the wrong sign, so it does
    not depend on how any particular optimiser build happens to terminate.

    **The other route to the same status -- a shallow decreasing curve chosen to make the
    optimiser give up -- is hardware-dependent, so it is not used.** Such a curve returned
    ``NOT_CONVERGED`` on macOS/arm64 and ``OK`` on Linux/x86_64 with byte-identical dependency
    versions (numpy 2.4.6, scipy 1.17.1, statsmodels 0.15.0). Constructing a status out of an
    optimiser's give-up point is not a reproducible test; constructing it out of a documented
    boundary check is.

    Every other ``NOT_CONVERGED`` test in the suite (``tests/unit/test_models_fit.py``) already
    uses the wrong-sign route, which is why none of them was affected.
    """

    def test_threshold_mirrors_not_converged_with_grid_attached(self) -> None:
        axis = Axis(name="wrong_sign_probe", unit="unit", scale="log")
        obs = Observations.from_counts(
            axis,
            severity=[1.0, 2.0, 4.0, 8.0],
            successes=[20, 35, 50, 65],
            trials=[100, 100, 100, 100],
            outcome="success",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)
        assert fit.status is Status.NOT_CONVERGED
        assert any("no interior maximum" in w for w in fit.warnings)

        # No control cell, so `FAILS_AT_BASELINE` cannot fire and the fallback is reached for
        # any target. `absolute` needs no asymptote, which a NOT_CONVERGED fit cannot supply
        # for a fit with no fitted parameters.
        t = threshold(
            fit, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL
        )

        assert t.status is fit.status is Status.NOT_CONVERGED
        assert t.value is None
        assert t.interval_method == "exact_bound"
        assert t.censoring in (Censoring.NONE, Censoring.RIGHT, Censoring.LEFT)
        # grid is set on every SEPARATION/NOT_CONVERGED exact-bound fallback.
        assert t.grid is not None

    def test_shallow_decreasing_probe_is_handled_consistently_whatever_status_it_gets(
        self,
    ) -> None:
        """A shallow decreasing dataset, used for the code paths it exercises -- the likelihood
        path with an estimated upper asymptote and a zero-severity control -- without a
        platform-dependent assertion on its status.

        Whether this fit converges is genuinely build-dependent: it returned ``NOT_CONVERGED``
        on macOS/arm64 and ``OK`` on Linux/x86_64 with identical dependency versions. Both are
        legitimate answers for a shallow curve near the edge of identifiability, and it is not
        robustmargin's job to make an optimiser terminate identically on every CPU. What *is*
        robustmargin's job is that the threshold agrees with whatever the fit decided, and that is
        what this asserts -- so the test exercises the path on both platforms and can only fail
        for a reason worth knowing about.
        """
        axis = Axis(name="shallow_probe", unit="unit", scale="log")
        obs = Observations.from_counts(
            axis,
            severity=[0.0, 1.0, 2.0, 4.0],
            successes=[40, 35, 25, 10],
            trials=[100, 100, 100, 100],
            outcome="success",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)

        # 0.3 clears the control's own one-sided upper bound (40/100 gives 0.4870 at 0.95), so
        # FAILS_AT_BASELINE does not short-circuit ahead of the path under test (the baseline
        # check comes first).
        t = threshold(
            fit, definition=Definition.absolute(0.3), interval_method="profile", level=_LEVEL
        )

        if fit.status is Status.OK:
            # A converged fit either reports a value, or says why it does not with a censoring
            # label that carries a bound.
            assert t.status in (Status.OK, Status.UNREACHABLE, Status.FAILS_AT_BASELINE)
            if t.status is Status.OK and t.censoring is Censoring.NONE:
                assert t.value is not None
        else:
            # A failed fit never produces a number, mirrors its status, and takes the
            # exact-bound fallback with the grid statistic attached.
            assert t.status is fit.status
            assert t.value is None
            assert t.interval_method == "exact_bound"
            assert t.grid is not None


class TestUnreachableAboveAFixedCeiling:
    """``absolute(a)`` with ``a`` at or beyond the fitted ``upper`` asymptote (reachability
    requires ``l < P* < u`` for *every* definition, not only ``absolute``). Uses a
    ``upper=0.9`` **fixed** (rather than estimated) ceiling: the reachability check
    reads ``fit.params["upper"].value`` the same way regardless of whether that parameter was
    fixed or estimated, so a fixed ceiling exercises the identical branch far more
    deterministically than coaxing a specific estimated value out of an optimizer, while control
    still clears the target easily (so this is not also ``FAILS_AT_BASELINE`` -- see
    ``TestFailsAtBaselineWinsOverUnreachable`` below for that combination).
    """

    def test_absolute_0_95_is_unreachable_when_upper_is_fixed_at_0_9(self) -> None:
        axis = Axis(name="unreachable_probe", unit="unit", scale="log")
        obs = Observations.from_counts(
            axis,
            severity=[0.0, 0.5, 1.0, 2.0],
            successes=[100, 70, 50, 10],
            trials=[100, 100, 100, 100],
            outcome="success",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=0.9, lower=0.0)
        assert fit.status is Status.OK
        assert fit.params is not None
        assert fit.params["upper"].value == 0.9
        assert fit.params["upper"].fixed is True

        t = threshold(
            fit, definition=Definition.absolute(0.95), interval_method="profile", level=_LEVEL
        )

        assert t.status is Status.UNREACHABLE
        assert t.censoring is Censoring.NONE
        assert t.value is None and t.lo is None and t.hi is None
        assert t.interval_method == "exact_bound"
        assert t.baseline is None
        assert t.grid is None


class TestUnreachableBaselineFractionBelowTheFloor:
    """``baseline_fraction(p)`` is unreachable whenever ``p * u <= l`` -- the
    target fraction of the control's own performance sits at or below the fitted floor. Uses
    both asymptotes fixed (``upper=0.9, lower=0.2``) for the same determinism reason as
    ``TestUnreachableAboveAFixedCeiling``: ``p=0.1`` gives a target of ``0.1 * 0.9 = 0.09``,
    below the fixed floor of ``0.2``.
    """

    def test_baseline_fraction_0_1_is_unreachable_when_p_times_u_is_at_or_below_l(self) -> None:
        axis = Axis(name="baseline_fraction_unreachable_probe", unit="unit", scale="log")
        obs = Observations.from_counts(
            axis,
            severity=[0.0, 0.5, 1.0, 2.0],
            successes=[90, 70, 50, 25],
            trials=[100, 100, 100, 100],
            outcome="success",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=0.9, lower=0.2)
        assert fit.status is Status.OK
        assert fit.params is not None
        assert fit.params["lower"].value == 0.2
        assert fit.params["upper"].value == 0.9

        t = threshold(
            fit,
            definition=Definition.baseline_fraction(0.1),
            interval_method="profile",
            level=_LEVEL,
        )

        assert t.status is Status.UNREACHABLE
        assert t.censoring is Censoring.NONE
        assert t.value is None and t.lo is None and t.hi is None
        assert t.interval_method == "exact_bound"


class TestFailsAtBaselineWinsOverUnreachable:
    """Ordering regression test: when the control already fails the criterion *and* the
    (independently, genuinely fitted) asymptote also cannot reach it, ``FAILS_AT_BASELINE``
    wins -- it rests on the sharper, exact test of the raw control cell, not on the fitted
    asymptote.

    Uses the committed ``tools/drc_reference/estimated_asymptotes.json`` simulation (true
    ``upper=0.9``, control 365/400) rather than any series in ``tests/data/pooled_counts.csv``:
    every ``axis_1`` series' ``upper="estimate"`` fit there is ``Status.SEPARATION`` (confirmed:
    ``tests/unit/test_models_pooled_counts.py``'s own ``..._upper_estimate_is_separation`` tests,
    for every one of ``system_a``/``system_b``/``system_c``), and the reachability check *cannot
    run* on a fit with no fitted parameters at all (it needs l and u, so it cannot run on the
    fallback path) -- so no series there can exercise the actual UNREACHABLE-vs- FAILS_AT_BASELINE
    conflict this test targets. This fixture's simulated data genuinely converges (confirmed
    empirically: ``fit_dose_response`` on it with ``upper="estimate", lower=0.0`` gives
    ``Status.OK`` with ``upper`` estimated at about ``0.928``), and its control (365/400 = 0.9125)
    has an exact one-sided 95% upper bound of about ``0.935`` -- below ``0.95`` -- so
    ``absolute(0.95)`` is *both* ``FAILS_AT_BASELINE`` (the control fails outright) *and* would
    independently be ``UNREACHABLE`` (the fitted ceiling, ~0.928, also cannot reach 0.95).
    """

    def test_absolute_0_95_is_fails_at_baseline_not_unreachable(self) -> None:
        with (_FIXTURE_DIR / "estimated_asymptotes.json").open() as fh:
            fixture = json.load(fh)
        sim = fixture["simulation"]
        axis = Axis(name="severity", unit="unit", scale="log")
        obs = Observations.from_counts(
            axis,
            severity=sim["severity"],
            successes=sim["successes"],
            trials=sim["trials"],
            outcome="success",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)
        assert fit.status is Status.OK
        assert fit.params is not None
        # Both preconditions for the ordering conflict, made explicit rather than assumed.
        assert fit.params["upper"].value < 0.95, (
            "fixture drifted: this test needs the fitted ceiling below the 0.95 criterion"
        )

        t = threshold(
            fit, definition=Definition.absolute(0.95), interval_method="profile", level=_LEVEL
        )

        assert t.status is Status.FAILS_AT_BASELINE
        assert t.status is not Status.UNREACHABLE
        assert t.baseline is not None
        assert t.baseline.successes == 365
        assert t.baseline.trials == 400


class TestOpenSidedProfileCensoring:
    """``OPEN_UPPER``/``OPEN_LOWER``: the value lies inside the tested range, but one side of its
    profile interval does not close within the search limit (``log(100)`` beyond the tested
    range). Tried as a small set of weakly-identified candidate designs (two close, well-sampled
    nonzero levels anchoring the crossing tightly on one side, plus a third, very-low-sample
    level placed far out on the other) rather than one specific number, because the exact
    numeric boundary of "does not close" depends on ``threshold()``'s own root-finding.

    If none of these candidates trigger ``OPEN_UPPER``/``OPEN_LOWER``, that is a finding about
    *this test's* construction -- replace the candidate list with one searched against the real
    implementation, rather than deleting the coverage requirement.
    """

    _CANDIDATES: tuple[tuple[list[float], list[int], list[int]], ...] = (
        ([0.0, 1.0, 1.5, 500.0], [100, 90, 40, 1], [100, 100, 100, 2]),
        ([0.0, 1.0, 1.2, 800.0], [100, 95, 30, 0], [100, 100, 100, 1]),
        ([0.0, 2.0, 2.5, 1000.0], [100, 85, 45, 1], [100, 100, 100, 3]),
    )

    def test_at_least_one_candidate_design_leaves_one_profile_side_open(self) -> None:
        axis = Axis(name="open_interval_probe", unit="unit", scale="log")
        found: Censoring | None = None

        for severity, successes, trials in self._CANDIDATES:
            obs = Observations.from_counts(
                axis,
                severity=severity,
                successes=successes,
                trials=trials,
                outcome="success",
                direction="decreasing",
            )
            fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
            if fit.status is not Status.OK:
                continue
            t = threshold(
                fit,
                definition=Definition.absolute(0.5),
                interval_method="profile",
                level=_LEVEL,
            )
            if t.censoring in (Censoring.OPEN_UPPER, Censoring.OPEN_LOWER):
                found = t.censoring
                if found is Censoring.OPEN_UPPER:
                    assert t.value is not None and t.lo is not None and t.hi is None
                else:
                    assert t.value is not None and t.hi is not None and t.lo is None
                assert t.status is Status.OK
                break

        assert found is not None, (
            "none of the constructed candidate designs produced OPEN_UPPER/OPEN_LOWER against "
            "the real threshold() implementation; this test's candidate list needs revisiting "
            "(the search-limit behaviour was unverified when the candidates were chosen)"
        )


class TestNonMonotoneContradictionWarningOnAConvergedFit:
    """The ``NON_MONOTONE: `` warning (token frozen) ends with the fixed phrase "no bounds
    are reported". That is true on the ``SEPARATION``/``NOT_CONVERGED`` fallback branch (see
    ``tests/unit/test_threshold_pooled_counts.py::TestSystemCAxis1NonMonotoneContradiction``, where
    ``Threshold.value``/``.lo``/``.hi`` really are all ``None``), but the same warning text is
    also used on a **converged** fit (``censoring.py``'s ``_contradiction_or_none``, the
    ``fit.status is Status.OK`` branch): there, the contradiction only suppresses the
    ``RIGHT``/``LEFT`` directional short-circuit, and ``threshold()`` still solves for ``value``
    and still runs the requested interval method, so ``Threshold.lo``/``.hi`` are typically set.
    "No bounds are reported" is then false of the very object carrying the warning -- only the
    *exact* one-sided bounds (the ``RIGHT``/``LEFT`` claim) go unreported, not the profile/delta
    ones.

    The data: the ``form_b`` curve from the `selenium` dataset in ``README.md``'s own worked
    example (already executed, with its own assertions, by ``tests/unit/
    test_readme_examples.py``, so this fit's convergence is independently exercised elsewhere --
    not a fixture invented for this test alone). Survival rises from severity 300 to 400 before
    falling again, and at ``target=0.5`` the exact one-sided bounds (confirmed below, at
    ``level=0.95``) put severity 300's upper bound (~0.483) below the target while severity
    400's lower bound (~0.528), a *higher* severity, sits above it -- a contradiction of
    the monotonicity assumption, on a fit that otherwise converges cleanly.
    """

    _CONC = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]
    _TOTAL = [141, 153, 142, 139, 154, 155]
    _DEAD = [2, 30, 59, 82, 62, 85]
    _TARGET = 0.5

    def _fit(self) -> Fit:
        axis = Axis(name="concentration", unit="ug/L", scale="log")
        successes = [n - d for n, d in zip(self._TOTAL, self._DEAD, strict=True)]
        obs = Observations.from_counts(
            axis,
            severity=self._CONC,
            successes=successes,
            trials=self._TOTAL,
            outcome="survival",
            direction="decreasing",
        )
        return fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)

    def test_fixture_premise_fit_converges_and_the_exact_bounds_contradict(self) -> None:
        fit = self._fit()
        assert fit.status is Status.OK

        # Severity 300 (index 3) confidently fails the target; severity 400 (index 4), a
        # *higher* severity, confidently passes it -- backwards from what a monotone decreasing
        # curve allows, and the premise this test's contradiction depends on.
        successes = [n - d for n, d in zip(self._TOTAL, self._DEAD, strict=True)]
        upper_300 = exact_one_sided_bound(successes[3], self._TOTAL[3], level=_LEVEL, side="upper")
        lower_400 = exact_one_sided_bound(successes[4], self._TOTAL[4], level=_LEVEL, side="lower")
        assert upper_300 < self._TARGET < lower_400

    def test_warning_starts_with_the_non_monotone_token_and_bounds_are_set(self) -> None:
        fit = self._fit()
        t = threshold(
            fit,
            definition=Definition.absolute(self._TARGET),
            interval_method="profile",
            level=_LEVEL,
        )

        assert t.status is Status.OK
        assert t.censoring is Censoring.NONE
        # The whole point: on this branch the solve path still ran, so there genuinely are
        # bounds -- unlike the SEPARATION-fallback case this token also covers.
        assert t.value is not None
        assert t.lo is not None
        assert t.hi is not None

        non_monotone = [w for w in t.warnings if w.startswith("NON_MONOTONE: ")]
        assert len(non_monotone) == 1, (
            f"expected exactly one NON_MONOTONE: warning, got {t.warnings!r}"
        )

    def test_warning_does_not_claim_no_bounds_are_reported_when_bounds_are_set(self) -> None:
        """The actual bug pin: ``t.lo``/``t.hi`` are not ``None`` (asserted above), so a warning
        claiming "no bounds are reported" is false of this ``Threshold``. The fixed wording must
        make clear that it is specifically the *exact* one-sided bounds that go unreported, not
        the profile/delta ones this object actually carries. Wording beyond that is
        deliberately not pinned here -- only the contradiction and the word "exact"."""
        fit = self._fit()
        t = threshold(
            fit,
            definition=Definition.absolute(self._TARGET),
            interval_method="profile",
            level=_LEVEL,
        )
        assert t.lo is not None and t.hi is not None

        non_monotone = next(w for w in t.warnings if w.startswith("NON_MONOTONE: "))
        assert "no bounds are reported" not in non_monotone, (
            "self-contradictory: this Threshold's lo/hi are set, but the warning claims no "
            f"bounds are reported: {non_monotone!r}"
        )
        assert "exact" in non_monotone, (
            "the warning should clarify that it is the exact one-sided bounds (the "
            f"RIGHT/LEFT claim), not the profile/delta ones, that go unreported: {non_monotone!r}"
        )
