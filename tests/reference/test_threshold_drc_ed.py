"""drc ``ED(..., interval="delta")`` parity for ``threshold()``.

**Reads only the committed ``tools/drc_reference/finney71.json``. Never runs R.**

Two things this file pins down.

1. **The three definitions coincide when ``u=1, l=0`` exactly.** ``finney71.json``'s ``mapping``
   block records that finney71 is fit on ``affected/total`` (failure, increasing with dose), so
   robustmargin fits the success orientation ``(total-affected)/total`` with ``upper=1.0,
   lower=0.0``, ``direction="decreasing"``. drc's ``ED(m, p)`` on the *affected* scale is
   therefore robustmargin's ``Definition.relative(p/100)``: with ``q = p/100``,
   the target success performance is ``1 - q``, which is *also* what
   ``Definition.absolute(1-q)`` and ``Definition.baseline_fraction(1-q)`` solve for, since
   ``u=1``. All three must therefore give the identical ``value``/``lo``/``hi``, not merely
   agree with drc.

2. **A definitional mismatch, written into this file rather than absorbed into a tolerance.** drc's
   ``ED(..., interval="delta")`` computes its ``Lower``/``Upper`` columns as ``Estimate +/-
   qnorm(0.975) * Std. Error`` directly on the **original dose scale** -- confirmed against
   every one of the six rows in ``finney71.json`` (``LN.2`` and ``LL.2``, each at ED10/50/90).
   Worked example, ``LN.2`` ED50: ``4.84550166 +/- 1.959964 * 0.24653956 = [4.362293,
   5.328710]``, which matches the fixture's ``Lower``/``Upper`` to better than ``1e-6``
   relative. The log-scale-exponentiated reconstruction (``estimate * exp(+/- z * SE /
   estimate)``) does **not** match any of the six rows -- it is off by roughly 2% at ED10 and
   0.5% at ED50, both far outside the 1e-3 endpoint tolerance.

   robustmargin's own delta method is defined on the **log(severity)** scale and then
   exponentiated, specifically so the interval stays positive. This is precisely the mismatch
   "drc computing delta intervals on the original scale rather than the log scale", and it is
   written into this fixture-reading test, not absorbed into a wider tolerance. So this file
   asserts:

   - the point estimate (``Threshold.value``) against drc's ``Estimate`` column, at the
     ordinary ``1e-4`` relative -- the ED value itself does not depend on which scale its
     interval happens to be built on;
   - a **reconstructed original-scale standard error**, ``value * sigma_theta`` where
     ``sigma_theta`` is robustmargin's own log-scale delta-method standard error (recovered from
     its reported ``lo``/``hi`` via ``sigma_theta = ln(hi / lo) / (2 * z)``, the exact inverse
     of the log-scale-then-exponentiate construction), against drc's
     own ``Std. Error`` column, at ``1e-3`` relative;
   - and **explicitly that robustmargin's ``lo``/``hi`` do not match drc's ``Lower``/``Upper``**
     beyond the usual endpoint tolerance, so a future change that silently makes the two
     conventions agree (which would itself mean the log-scale convention was dropped, a
     regression) is caught rather than passing unnoticed.
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
    Definition,
    Observations,
    Status,
    Threshold,
    fit_dose_response,
    threshold,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tools" / "drc_reference"
_LEVEL = 0.95
_REL_TOL_VALUE = 1e-4
_REL_TOL_SE = 1e-3
# The ordinary endpoint tolerance: used here to prove robustmargin's endpoints fall *outside* it
# against drc's original-scale endpoints, not inside it.
_REL_TOL_ENDPOINTS = 1e-3


def _load(name: str) -> dict[str, Any]:
    with (_FIXTURE_DIR / name).open() as fh:
        result: dict[str, Any] = json.load(fh)
    return result


def _finney71_observations() -> Observations:
    fixture = _load("finney71.json")
    dose = fixture["dataset"]["dose"]
    total = fixture["dataset"]["total"]
    affected = fixture["dataset"]["affected"]
    successes = [t - a for t, a in zip(total, affected, strict=True)]
    axis = Axis(name="dose", unit="unit", scale="log")
    return Observations.from_counts(
        axis,
        severity=dose,
        successes=successes,
        trials=total,
        outcome="success",
        direction="decreasing",
    )


def _reconstructed_original_scale_se(t: Threshold) -> float:
    """Invert the log-scale-then-exponentiate delta construction: given
    robustmargin's own ``value``/``lo``/``hi``, recover the log-scale standard error
    ``sigma_theta`` it must have used (``lo = value * exp(-z*sigma_theta)``, ``hi = value *
    exp(+z*sigma_theta)``), then propagate it onto the original (dose) scale via
    ``se_original = value * sigma_theta`` -- the standard first-order delta-method identity
    ``d(exp(theta))/d(theta) = exp(theta)`` evaluated at ``theta = log(value)``, which is what
    makes this reconstruction the quantity to compare against drc's own original-scale
    ``Std. Error`` (not against drc's ``Lower``/``Upper`` directly -- see the module docstring).
    """
    assert t.value is not None and t.lo is not None and t.hi is not None
    z = float(norm.ppf(0.5 + _LEVEL / 2.0))
    sigma_theta = math.log(t.hi / t.lo) / (2.0 * z)
    return t.value * sigma_theta


@pytest.mark.parametrize(("link", "drc_key"), [("probit", "LN2"), ("logit", "LL2")])
class TestEdDeltaParity:
    """finney71, fixed ``upper=1.0, lower=0.0`` -- against drc's ``ED(..., interval="delta")``
    for ``LN.2`` (probit) and ``LL.2`` (logit), at ``p in {10, 50, 90}``.
    """

    def test_value_matches_drc_estimate_and_the_three_definitions_agree(
        self, link: str, drc_key: str
    ) -> None:
        fixture = _load("finney71.json")
        ed_delta = fixture["drc"][drc_key]["ED_delta"]

        obs = _finney71_observations()
        fit = fit_dose_response(obs, model="binomial", link=link, upper=1.0, lower=0.0)
        assert fit.status is Status.OK

        for p, row in zip(ed_delta["p"], ed_delta["table"]["values"], strict=True):
            estimate, _se, _lo, _hi = row
            q = p / 100.0
            target = 1.0 - q

            t_absolute = threshold(
                fit,
                definition=Definition.absolute(target),
                interval_method="delta",
                level=_LEVEL,
            )
            t_baseline = threshold(
                fit,
                definition=Definition.baseline_fraction(target),
                interval_method="delta",
                level=_LEVEL,
            )
            t_relative = threshold(
                fit, definition=Definition.relative(q), interval_method="delta", level=_LEVEL
            )

            assert t_absolute.value == pytest.approx(estimate, rel=_REL_TOL_VALUE)
            # u=1, l=0 exactly: all three definitions solve for the same target performance
            # (module docstring, point 1), so they must produce the identical threshold, not
            # merely three separately-correct ones.
            assert t_baseline.value == pytest.approx(t_absolute.value, rel=1e-12)
            assert t_relative.value == pytest.approx(t_absolute.value, rel=1e-12)
            assert t_baseline.lo == pytest.approx(t_absolute.lo, rel=1e-12)
            assert t_relative.lo == pytest.approx(t_absolute.lo, rel=1e-12)
            assert t_baseline.hi == pytest.approx(t_absolute.hi, rel=1e-12)
            assert t_relative.hi == pytest.approx(t_absolute.hi, rel=1e-12)

    def test_delta_se_matches_drc_on_the_original_scale(self, link: str, drc_key: str) -> None:
        fixture = _load("finney71.json")
        ed_delta = fixture["drc"][drc_key]["ED_delta"]

        obs = _finney71_observations()
        fit = fit_dose_response(obs, model="binomial", link=link, upper=1.0, lower=0.0)

        for p, row in zip(ed_delta["p"], ed_delta["table"]["values"], strict=True):
            _estimate, se, _lo, _hi = row
            target = 1.0 - p / 100.0

            t = threshold(
                fit,
                definition=Definition.absolute(target),
                interval_method="delta",
                level=_LEVEL,
            )

            reconstructed_se = _reconstructed_original_scale_se(t)
            assert reconstructed_se == pytest.approx(se, rel=_REL_TOL_SE)

    def test_delta_endpoints_do_not_match_drcs_original_scale_endpoints(
        self, link: str, drc_key: str
    ) -> None:
        """The definitional mismatch, pinned as a regression guard: if this ever starts passing
        at the ordinary endpoint tolerance, robustmargin's delta method silently stopped
        being computed on the log scale (module docstring, point 2) -- that is a finding to
        report, not a reason to delete this test.
        """
        fixture = _load("finney71.json")
        ed_delta = fixture["drc"][drc_key]["ED_delta"]

        obs = _finney71_observations()
        fit = fit_dose_response(obs, model="binomial", link=link, upper=1.0, lower=0.0)

        for p, row in zip(ed_delta["p"], ed_delta["table"]["values"], strict=True):
            _estimate, _se, lo, hi = row
            target = 1.0 - p / 100.0

            t = threshold(
                fit,
                definition=Definition.absolute(target),
                interval_method="delta",
                level=_LEVEL,
            )

            assert t.lo is not None and t.hi is not None
            assert not math.isclose(t.lo, lo, rel_tol=_REL_TOL_ENDPOINTS)
            assert not math.isclose(t.hi, hi, rel_tol=_REL_TOL_ENDPOINTS)
