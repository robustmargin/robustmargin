"""drc ``EDcomp`` Fieller/delta parity for ``ratio_interval()``.

**Reads only the committed ``tools/drc_reference/selenium_edcomp.json``. Never runs R.**

The fixture fits drc's ``LL.2`` (2-parameter log-logistic, fixed asymptotes ``c=0, d=1`` on the
*affected* scale) with ``curveid = type`` across four selenium toxicity curves, then runs
``EDcomp(m, c(50, 50), interval = "fieller")`` and ``interval = "delta"`` on every pair -- six
pairs of ED50 ratios in total. This file rebuilds the same four curves as robustmargin
:class:`~robustmargin.Fit`/:class:`~robustmargin.Threshold` objects (success-oriented, fixed
``upper=1.0, lower=0.0``, ``absolute(0.5)``) and compares ``ratio_interval()`` against both
tables.

**A data-compatibility note this file works around, not silently.** Every one of the four curves'
``conc=0`` control rows has nonzero deaths (2-3%). ``fit_dose_response`` -- and
:class:`~robustmargin.Fit` itself, independent of the fitting routine -- both refuse to build a
``upper=1.0``-fixed fit against a failing control on a log axis (``Status.CONTROL_INCOMPATIBLE``):
the fixed asymptote says the likelihood of any control failure is exactly zero, and robustmargin has
no silent fallbacks, so it raises rather than clipping the offending probability away from zero the
way drc's own fit evidently tolerates. This file therefore fits robustmargin's model on the
non-zero-dose rows only for every curve -- the same set of rows drc's own ``LL.2`` GLM equivalent
uses as covariate data (dropping the always-``z*=0``-at-``dose=0`` control row from the fitted
regression is exactly what ``models.py``'s own fixed-asymptote GLM path already does for a
*compatible* control, ``_fit_glm``'s ``fit_cells``) -- and confirms below
(``TestSeleniumFitMatchesDrcCoefficients``) that doing so reproduces drc's own ``(b, e)`` to the
ordinary ``1e-4`` relative tolerance. That agreement is what licenses comparing the resulting ratio
against drc's ``EDcomp`` tables at all: this is the same statistical model as drc's, fit on the same
informative data, not a different one.

**Two definitional mismatches found while building this file, written in here rather than
absorbed into a tolerance:**

1. **``EDcomp(..., interval="fieller")`` uses a Student-t quantile, not a normal one.** Every
   one of the six pairs' ``Lower``/``Upper`` bounds reproduces exactly (to floating-point
   precision) from the classic Fieller quadratic using ``qt(0.975, df=17)``
   in place of ``qnorm(0.975)`` -- confirmed against drc's own reported ``e``/``vcov`` entries
   directly, independent of anything robustmargin computes (this file's
   ``test_reconstructed_fieller_matches_drc_with_drcs_own_t_quantile``). ``df=17`` is
   ``25 - 8``: 25 total observations across the four curves, 8 free parameters (``b``, ``e`` per
   curve). robustmargin's whole interval framework is asymptotic-normal (``chi2``/``norm`` only --
   confirmed by inspection, there is no ``t``-distribution or degrees-of-freedom concept
   anywhere in ``Fit``/``Threshold``/``Ratio``), so ``ratio_interval(method="fieller")`` cannot
   reproduce this even in principle. The point **estimate** (the ratio itself) does not depend
   on the quantile and matches drc exactly; only the endpoints are affected, and are asserted
   below to *not* match at the usual endpoint tolerance, as a regression guard against
   this difference ever silently disappearing (which would mean a t-quantile had been added, or
   drc's own default had changed, either of which is worth knowing about).
2. **``EDcomp(..., interval="delta")`` is a plain Wald interval on the ratio's own original
   (linear) scale, ``Estimate +/- qnorm(0.975) * SE``, not robustmargin's
   log-scale-then-exponentiate construction.** This is the same "drc computes delta intervals
   on the original scale rather than the log scale" mismatch
   ``tests/reference/test_threshold_drc_ed.py`` already documents for a single threshold's
   ``ED``, now confirmed for ``EDcomp`` too (normal quantile this time, not ``t`` -- the two
   ``EDcomp`` interval types use different quantiles from each other, which is itself worth
   recording precisely because it looks like it should not be true).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from scipy.stats import norm
from scipy.stats import t as student_t

from robustmargin import (
    Axis,
    Definition,
    IntervalShape,
    Observations,
    Status,
    Threshold,
    fit_dose_response,
    ratio_interval,
    threshold,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tools" / "drc_reference"
_LEVEL = 0.95
_Z = float(norm.ppf(0.5 + _LEVEL / 2.0))
_REL_TOL_PARAMS = 1e-4
_REL_TOL_SE = 1e-3
_REL_TOL_ENDPOINTS = 1e-3
# 25 observations (6 + 6 + 8 + 5 rows across the four curves) minus 8 free parameters (b, e per
# curve): drc's own residual degrees of freedom for this joint curveid fit.
_RESIDUAL_DF = 17
_PAIRS: tuple[tuple[int, int], ...] = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
_AXIS = Axis(name="conc", unit="unit", scale="log")


def _load(name: str) -> dict[str, Any]:
    with (_FIXTURE_DIR / name).open() as fh:
        result: dict[str, Any] = json.load(fh)
    return result


def _selenium_observations_by_type() -> dict[int, Observations]:
    fixture = _load("selenium_edcomp.json")
    dataset = fixture["dataset"]
    by_type: dict[int, list[tuple[float, int, int]]] = {}
    for curve_type, conc, total, dead in zip(
        dataset["type"], dataset["conc"], dataset["total"], dataset["dead"], strict=True
    ):
        if conc == 0.0:
            # Excluded deliberately -- see module docstring's data-compatibility note.
            continue
        by_type.setdefault(curve_type, []).append((float(conc), int(total - dead), int(total)))

    observations: dict[int, Observations] = {}
    for curve_type, rows in by_type.items():
        observations[curve_type] = Observations.from_counts(
            _AXIS,
            severity=[r[0] for r in rows],
            successes=[r[1] for r in rows],
            trials=[r[2] for r in rows],
            outcome="success",
            direction="decreasing",
        )
    return observations


def _selenium_fits() -> dict[int, Any]:
    fits = {}
    for curve_type, obs in _selenium_observations_by_type().items():
        fit = fit_dose_response(obs, model="binomial", link="logit", upper=1.0, lower=0.0)
        assert fit.status is Status.OK
        fits[curve_type] = fit
    return fits


def _selenium_thresholds(*, interval_method: str) -> dict[int, Threshold]:
    return {
        curve_type: threshold(
            fit, definition=Definition.absolute(0.5), interval_method=interval_method, level=_LEVEL
        )
        for curve_type, fit in _selenium_fits().items()
    }


def _drc_coef() -> dict[str, float]:
    coef: dict[str, float] = _load("selenium_edcomp.json")["model"]["coef"]
    return coef


def _drc_var_e(curve_type: int) -> float:
    """``Var(e:i)`` straight from drc's own ``vcov``, used only for the independent
    ``t``-quantile reconstruction check, never for the robustmargin-facing assertions."""
    fixture = _load("selenium_edcomp.json")
    vcov = fixture["model"]["vcov"]
    i = vcov["row_names"].index(f"e:{curve_type}")
    value: float = vcov["values"][i][i]
    return value


def _fieller_quadratic(
    *, theta_a: float, va: float, theta_b: float, vb: float, z: float
) -> tuple[float, float]:
    """The classic Fieller quadratic's two roots, assuming ``C=0``
    (independent) and ``A > 0`` (``g < 1``, the ``BOUNDED`` case every selenium pair falls into,
    checked directly below)."""
    a_coef = theta_b**2 - z**2 * vb
    b_coef = theta_a * theta_b
    k_coef = theta_a**2 - z**2 * va
    disc = b_coef**2 - a_coef * k_coef
    assert a_coef > 0.0 and disc > 0.0
    sq = math.sqrt(disc)
    lo, hi = sorted([(b_coef - sq) / a_coef, (b_coef + sq) / a_coef])
    return lo, hi


class TestSeleniumFitMatchesDrcCoefficients:
    """Licenses every other comparison in this file: fitting robustmargin's model on the non-zero
    dose rows alone reproduces drc's own ``LL.2`` ``(b, e)`` (module docstring).

    ``mu = log(e)``, ``s = -1/b``: derived directly from ``LL.2``'s documented closed form,
    ``F_dead(x) = 1 / (1 + exp(b*(log(x) - log(e))))``, the same sign convention
    ``tools/drc_reference/README.md`` already confirms for ``finney71``'s ``LL.2`` fit (also a
    failure-increasing-with-dose orientation) -- not assumed by re-use, but re-derivable from the
    formula: robustmargin's success-scale ``F(z) = sigma(z)`` at ``z = (log(x)-mu)/s`` must equal
    drc's own ``F_dead(x)`` (since ``P_success = 1 - F_dead`` and ``upper=1, lower=0`` fixed makes
    robustmargin's ``P_success = 1 - F(z)`` too), which forces ``z = -b*(log(x)-log(e))``, i.e.
    ``s = -1/b``, ``mu = log(e)``.
    """

    def test_mu_and_s_match_drc_b_e_derivation(self) -> None:
        coef = _drc_coef()
        for curve_type, fit in _selenium_fits().items():
            b = coef[f"b:{curve_type}"]
            e = coef[f"e:{curve_type}"]
            assert b < 0.0
            expected_mu = math.log(e)
            expected_s = -1.0 / b
            assert fit.params is not None
            assert fit.params["mu"].value == pytest.approx(expected_mu, rel=_REL_TOL_PARAMS)
            assert fit.params["s"].value == pytest.approx(expected_s, rel=_REL_TOL_PARAMS)

    def test_threshold_value_matches_drc_e_exactly(self) -> None:
        """``absolute(0.5)`` with ``upper=1, lower=0`` gives ``z*=0``, so
        ``theta_hat = mu`` and ``value = exp(mu) = e`` exactly -- no delta/profile interval
        machinery involved in the point estimate itself."""
        coef = _drc_coef()
        for curve_type, t in _selenium_thresholds(interval_method="delta").items():
            assert t.value == pytest.approx(coef[f"e:{curve_type}"], rel=_REL_TOL_PARAMS)


class TestEdcompFiellerParity:
    @pytest.mark.parametrize(("a", "b"), _PAIRS)
    def test_estimate_matches_drc(self, a: int, b: int) -> None:
        fixture = _load("selenium_edcomp.json")
        row = fixture["EDcomp_fieller"]["table"]["values"][_PAIRS.index((a, b))]
        thresholds = _selenium_thresholds(interval_method="profile")

        result = ratio_interval(
            thresholds[a], thresholds[b], method="fieller", dependence="independent", level=_LEVEL
        )

        assert result.estimate == pytest.approx(row[0], rel=_REL_TOL_PARAMS)

    @pytest.mark.parametrize(("a", "b"), _PAIRS)
    def test_shape_is_bounded(self, a: int, b: int) -> None:
        thresholds = _selenium_thresholds(interval_method="profile")

        result = ratio_interval(
            thresholds[a], thresholds[b], method="fieller", dependence="independent", level=_LEVEL
        )

        assert result.shape is IntervalShape.BOUNDED
        assert result.lo is not None and result.hi is not None
        assert result.lo <= result.estimate <= result.hi

    @pytest.mark.parametrize(("a", "b"), _PAIRS)
    def test_endpoints_do_not_match_drc_at_the_plans_usual_tolerance(self, a: int, b: int) -> None:
        """Regression guard for mismatch 1 (module docstring): if this starts passing, either a
        t-quantile crept into ``ratio_interval``, or drc's own default changed -- both worth a
        stop-and-look, not a silent green.
        """
        fixture = _load("selenium_edcomp.json")
        row = fixture["EDcomp_fieller"]["table"]["values"][_PAIRS.index((a, b))]
        _estimate, drc_lo, drc_hi = row
        thresholds = _selenium_thresholds(interval_method="profile")

        result = ratio_interval(
            thresholds[a], thresholds[b], method="fieller", dependence="independent", level=_LEVEL
        )

        assert result.lo is not None and result.hi is not None
        assert not math.isclose(result.lo, drc_lo, rel_tol=_REL_TOL_ENDPOINTS)
        assert not math.isclose(result.hi, drc_hi, rel_tol=_REL_TOL_ENDPOINTS)

    @pytest.mark.parametrize(("a", "b"), _PAIRS)
    def test_reconstructed_fieller_matches_drc_with_drcs_own_t_quantile(
        self, a: int, b: int
    ) -> None:
        """Independent check (mismatch 1, module docstring): using drc's *own* ``Var(e:i))``
        (not anything robustmargin computed) and ``qt(0.975, df=17)`` in the classic Fieller
        quadratic reproduces drc's ``Lower``/``Upper`` columns exactly. This isolates the
        quantile choice as the entire explanation for the mismatch above -- the underlying
        Fieller algebra is not in question.
        """
        fixture = _load("selenium_edcomp.json")
        row = fixture["EDcomp_fieller"]["table"]["values"][_PAIRS.index((a, b))]
        _estimate, drc_lo, drc_hi = row
        coef = _drc_coef()
        theta_a, theta_b = coef[f"e:{a}"], coef[f"e:{b}"]
        va, vb = _drc_var_e(a), _drc_var_e(b)
        t_quantile = float(student_t.ppf(0.975, _RESIDUAL_DF))

        lo, hi = _fieller_quadratic(theta_a=theta_a, va=va, theta_b=theta_b, vb=vb, z=t_quantile)

        assert lo == pytest.approx(drc_lo, rel=1e-6)
        assert hi == pytest.approx(drc_hi, rel=1e-6)


class TestEdcompDeltaParity:
    @pytest.mark.parametrize(("a", "b"), _PAIRS)
    def test_estimate_matches_drc(self, a: int, b: int) -> None:
        fixture = _load("selenium_edcomp.json")
        row = fixture["EDcomp_delta"]["table"]["values"][_PAIRS.index((a, b))]
        thresholds = _selenium_thresholds(interval_method="profile")

        result = ratio_interval(
            thresholds[a], thresholds[b], method="log_delta", dependence="independent", level=_LEVEL
        )

        assert result.estimate == pytest.approx(row[0], rel=_REL_TOL_PARAMS)

    @pytest.mark.parametrize(("a", "b"), _PAIRS)
    def test_endpoints_do_not_match_drc_at_the_plans_usual_tolerance(self, a: int, b: int) -> None:
        """Regression guard for mismatch 2 (module docstring)."""
        fixture = _load("selenium_edcomp.json")
        row = fixture["EDcomp_delta"]["table"]["values"][_PAIRS.index((a, b))]
        _estimate, drc_lo, drc_hi = row
        thresholds = _selenium_thresholds(interval_method="profile")

        result = ratio_interval(
            thresholds[a], thresholds[b], method="log_delta", dependence="independent", level=_LEVEL
        )

        assert result.lo is not None and result.hi is not None
        assert not math.isclose(result.lo, drc_lo, rel_tol=_REL_TOL_ENDPOINTS)
        assert not math.isclose(result.hi, drc_hi, rel_tol=_REL_TOL_ENDPOINTS)

    @pytest.mark.parametrize(("a", "b"), _PAIRS)
    def test_reconstructed_se_matches_drcs_own_original_scale_se(self, a: int, b: int) -> None:
        """Mirrors ``test_threshold_drc_ed.py``'s ``_reconstructed_original_scale_se``: invert
        robustmargin's log-scale-then-exponentiate construction to recover its own log-scale ratio
        standard error, propagate it onto the original (ratio) scale via
        ``se_original = estimate * sigma_log`` (the same delta-method identity this whole file's
        docstring cites), and compare to drc's own original-scale SE, recovered from
        ``EDcomp_delta``'s additive-symmetric ``Lower``/``Upper`` as
        ``(Upper - Lower) / (2 * qnorm(0.975))`` -- confirmed additive-symmetric (not
        log-symmetric) directly in the fixture: ``Upper - Estimate == Estimate - Lower`` to
        floating-point precision for every one of the six pairs.
        """
        fixture = _load("selenium_edcomp.json")
        row = fixture["EDcomp_delta"]["table"]["values"][_PAIRS.index((a, b))]
        estimate, drc_lo, drc_hi = row
        assert (estimate - drc_lo) == pytest.approx(drc_hi - estimate, rel=1e-6)
        drc_se = (drc_hi - drc_lo) / (2.0 * _Z)

        thresholds = _selenium_thresholds(interval_method="profile")
        result = ratio_interval(
            thresholds[a], thresholds[b], method="log_delta", dependence="independent", level=_LEVEL
        )
        assert result.lo is not None and result.hi is not None and result.estimate is not None

        sigma_log_ratio = math.log(result.hi / result.lo) / (2.0 * _Z)
        reconstructed_se = result.estimate * sigma_log_ratio

        assert reconstructed_se == pytest.approx(drc_se, rel=_REL_TOL_SE)
