"""drc parity for an **increasing** outcome, through the mirror.

**Reads only the committed JSON under ``tools/drc_reference/``. Never runs R. No new R run.**

drc's ``LL.2``/``LN.2`` fits on ``finney71`` and ``selenium`` are fit on the *affected* (``dead``)
counts, which rise with dose, with the asymptotes fixed at ``d = 1`` and ``c = 0``. That is exactly
the outcome an ``Observations(direction="increasing")`` describes, so here robustmargin is given
the **affected counts directly** and must reproduce drc's own numbers. The existing tests in
``test_threshold_drc_ed.py``, ``test_threshold_profile.py`` and ``test_ratio_drc_edcomp.py`` reach
the same numbers by fitting the complement (``total - affected``) as a decreasing success rate;
this file is the same fixtures read through the other orientation, so it also pins that the two
agree.

The mapping from drc's quantity to the robustmargin call is the mirror, written out so it is not
re-derived by the reader:

* ``ED(m, p)`` on the affected scale, with ``q = p / 100``, is the dose at which the affected rate
  reaches ``q``. Increasing ``Definition.absolute(q)`` solves for that. It mirrors to decreasing
  ``absolute(1 - q)`` on the complement, which is what ``test_threshold_drc_ed.py`` calls.
* ``Definition.relative(q)`` is direction-symmetric, and with ``u = 1, l = 0`` it is the same
  target as ``absolute(q)``.
* Asymptotes: drc's fixed ``d = 1, c = 0`` map to ``upper=1.0, lower=0.0`` on the outcome scale.
  The complement then has ``(u', l') = (1 - 0, 1 - 1) = (1, 0)`` as well, so nothing needs swapping
  for these fixtures.

**Definitional notes, carried over from the decreasing-orientation files (not absorbed into a
tolerance).** drc's ``ED(..., interval="delta")`` ``Lower``/``Upper`` are ``Estimate +/-
qnorm(0.975) * SE`` on the original dose scale, while robustmargin's delta interval is built on
``log(severity)`` and exponentiated, so only ``Estimate`` and ``Std. Error`` (after the same
reconstruction used there) are compared, and the endpoints are asserted *not* to match. Likewise
``EDcomp(interval="delta")`` is additive-symmetric on the ratio's own scale, and ``EDcomp``'s
Fieller interval uses a Student-t quantile, so only the ratio's estimate and reconstructed
standard error are compared.
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
    ratio_interval,
    threshold,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tools" / "drc_reference"
_LEVEL = 0.95
_Z = float(norm.ppf(0.5 + _LEVEL / 2.0))
_REL_TOL_VALUE = 1e-4
_REL_TOL_SE = 1e-3
_REL_TOL_ENDPOINTS = 1e-3
# The profile reference's point estimate comes from a converged GLM on both sides, as in
# test_threshold_profile.py, and is held to the same tight tolerance.
_REL_TOL_THETA_HAT = 1e-8
_PAIRS: tuple[tuple[int, int], ...] = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))


def _load(name: str) -> dict[str, Any]:
    with (_FIXTURE_DIR / name).open() as fh:
        result: dict[str, Any] = json.load(fh)
    return result


def _finney71_increasing_observations() -> Observations:
    fixture = _load("finney71.json")
    dataset = fixture["dataset"]
    axis = Axis(name="dose", unit="unit", scale="log")
    return Observations.from_counts(
        axis,
        severity=dataset["dose"],
        successes=dataset["affected"],
        trials=dataset["total"],
        outcome="affected",
        direction="increasing",
    )


def _reconstructed_original_scale_se(t: Threshold) -> float:
    """Invert the log-scale-then-exponentiate delta interval to its log-scale standard error and
    propagate it to the dose scale, as ``test_threshold_drc_ed.py`` does."""
    assert t.value is not None and t.lo is not None and t.hi is not None
    sigma_theta = math.log(t.hi / t.lo) / (2.0 * _Z)
    return t.value * sigma_theta


@pytest.mark.parametrize(("link", "drc_key"), [("probit", "LN2"), ("logit", "LL2")])
class TestFinney71IncreasingEdDelta:
    def test_value_matches_drc_estimate_for_absolute_and_relative(
        self, link: str, drc_key: str
    ) -> None:
        ed_delta = _load("finney71.json")["drc"][drc_key]["ED_delta"]
        fit = fit_dose_response(
            _finney71_increasing_observations(),
            model="binomial",
            link=link,
            upper=1.0,
            lower=0.0,
        )
        assert fit.status is Status.OK
        assert fit.direction == "increasing"

        for p, row in zip(ed_delta["p"], ed_delta["table"]["values"], strict=True):
            estimate = row[0]
            q = p / 100.0
            t_absolute = threshold(
                fit,
                definition=Definition.absolute(q),
                interval_method="delta",
                level=_LEVEL,
            )
            t_relative = threshold(
                fit, definition=Definition.relative(q), interval_method="delta", level=_LEVEL
            )

            assert t_absolute.direction == "increasing"
            assert t_absolute.value == pytest.approx(estimate, rel=_REL_TOL_VALUE)
            # u = 1, l = 0: relative(q) and absolute(q) are the same target.
            assert t_relative.value == pytest.approx(t_absolute.value, rel=1e-12)
            assert t_relative.lo == pytest.approx(t_absolute.lo, rel=1e-12)
            assert t_relative.hi == pytest.approx(t_absolute.hi, rel=1e-12)

    def test_delta_se_matches_drc_on_the_original_scale(self, link: str, drc_key: str) -> None:
        ed_delta = _load("finney71.json")["drc"][drc_key]["ED_delta"]
        fit = fit_dose_response(
            _finney71_increasing_observations(),
            model="binomial",
            link=link,
            upper=1.0,
            lower=0.0,
        )

        for p, row in zip(ed_delta["p"], ed_delta["table"]["values"], strict=True):
            se = row[1]
            t = threshold(
                fit,
                definition=Definition.absolute(p / 100.0),
                interval_method="delta",
                level=_LEVEL,
            )

            assert _reconstructed_original_scale_se(t) == pytest.approx(se, rel=_REL_TOL_SE)

    def test_delta_endpoints_do_not_match_drcs_original_scale_endpoints(
        self, link: str, drc_key: str
    ) -> None:
        """The definitional mismatch (module docstring), pinned as a guard: if the endpoints start
        to agree at the ordinary tolerance, the delta interval stopped being log-scale."""
        ed_delta = _load("finney71.json")["drc"][drc_key]["ED_delta"]
        fit = fit_dose_response(
            _finney71_increasing_observations(),
            model="binomial",
            link=link,
            upper=1.0,
            lower=0.0,
        )

        for p, row in zip(ed_delta["p"], ed_delta["table"]["values"], strict=True):
            drc_lo, drc_hi = row[2], row[3]
            t = threshold(
                fit,
                definition=Definition.absolute(p / 100.0),
                interval_method="delta",
                level=_LEVEL,
            )

            assert t.lo is not None and t.hi is not None
            assert not math.isclose(t.lo, drc_lo, rel_tol=_REL_TOL_ENDPOINTS)
            assert not math.isclose(t.hi, drc_hi, rel_tol=_REL_TOL_ENDPOINTS)


class TestFinney71IncreasingProfileAtEd50:
    def test_value_and_profile_endpoints_match_the_offset_trick_reference(self) -> None:
        """``finney71_profile.json`` is the independent R offset-trick profile at the ED50. Its
        endpoint tolerance is the fixture's own root-finding precision, as documented in
        ``test_threshold_profile.py``."""
        fixture = _load("finney71_profile.json")
        fit = fit_dose_response(
            _finney71_increasing_observations(),
            model="binomial",
            link="probit",
            upper=1.0,
            lower=0.0,
        )
        assert fit.status is Status.OK

        t = threshold(
            fit,
            definition=Definition.absolute(0.5),
            interval_method="profile",
            level=fixture["level"],
        )

        assert t.interval_method == "profile"
        assert t.value == pytest.approx(fixture["ed50_dose_hat"], rel=_REL_TOL_THETA_HAT)
        assert t.lo == pytest.approx(fixture["profile_ci_dose"]["lo"], rel=_REL_TOL_ENDPOINTS)
        assert t.hi == pytest.approx(fixture["profile_ci_dose"]["hi"], rel=_REL_TOL_ENDPOINTS)


def _selenium_increasing_thresholds() -> dict[int, Threshold]:
    """The four selenium curves fit on the ``dead`` counts (the outcome that rises with dose),
    nonzero doses only, for the data-compatibility reason ``test_ratio_drc_edcomp.py`` documents
    (every ``conc=0`` control has deaths, which a fixed ``lower=0`` refuses)."""
    dataset = _load("selenium_edcomp.json")["dataset"]
    axis = Axis(name="conc", unit="unit", scale="log")
    rows: dict[int, list[tuple[float, int, int]]] = {}
    for curve_type, conc, total, dead in zip(
        dataset["type"], dataset["conc"], dataset["total"], dataset["dead"], strict=True
    ):
        if conc == 0.0:
            continue
        rows.setdefault(curve_type, []).append((float(conc), int(dead), int(total)))

    out: dict[int, Threshold] = {}
    for curve_type, curve_rows in rows.items():
        obs = Observations.from_counts(
            axis,
            severity=[r[0] for r in curve_rows],
            successes=[r[1] for r in curve_rows],
            trials=[r[2] for r in curve_rows],
            outcome="dead",
            direction="increasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="logit", upper=1.0, lower=0.0)
        assert fit.status is Status.OK
        out[curve_type] = threshold(
            fit, definition=Definition.absolute(0.5), interval_method="profile", level=_LEVEL
        )
    return out


class TestSeleniumIncreasingEdcomp:
    def test_threshold_value_matches_drc_e(self) -> None:
        coef = _load("selenium_edcomp.json")["model"]["coef"]

        for curve_type, t in _selenium_increasing_thresholds().items():
            assert t.value == pytest.approx(coef[f"e:{curve_type}"], rel=_REL_TOL_VALUE)

    @pytest.mark.parametrize(("a", "b"), _PAIRS)
    def test_fieller_ratio_estimate_matches_drc(self, a: int, b: int) -> None:
        row = _load("selenium_edcomp.json")["EDcomp_fieller"]["table"]["values"][
            _PAIRS.index((a, b))
        ]
        thresholds = _selenium_increasing_thresholds()

        result = ratio_interval(
            thresholds[a], thresholds[b], method="fieller", dependence="independent", level=_LEVEL
        )

        assert result.status is Status.OK
        assert result.estimate == pytest.approx(row[0], rel=_REL_TOL_VALUE)

    @pytest.mark.parametrize(("a", "b"), _PAIRS)
    def test_log_delta_ratio_se_matches_drcs_original_scale_se(self, a: int, b: int) -> None:
        row = _load("selenium_edcomp.json")["EDcomp_delta"]["table"]["values"][_PAIRS.index((a, b))]
        estimate, drc_lo, drc_hi = row
        drc_se = (drc_hi - drc_lo) / (2.0 * _Z)
        thresholds = _selenium_increasing_thresholds()

        result = ratio_interval(
            thresholds[a], thresholds[b], method="log_delta", dependence="independent", level=_LEVEL
        )

        assert result.lo is not None and result.hi is not None and result.estimate is not None
        assert result.estimate == pytest.approx(estimate, rel=_REL_TOL_VALUE)
        reconstructed_se = result.estimate * math.log(result.hi / result.lo) / (2.0 * _Z)
        assert reconstructed_se == pytest.approx(drc_se, rel=_REL_TOL_SE)
