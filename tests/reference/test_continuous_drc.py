"""drc ``LL.4`` / ``LN.4`` parity for Gaussian (continuous-outcome) fits.

**Reads only the committed ``tools/drc_reference/ryegrass_ll4.json``. Never runs R.** A missing
fixture fails the test with ``FileNotFoundError``, as for every other reference file.

The logit link is drc's ``LL.4`` and the probit link is ``LN.4``. The fixture's ``mapped`` block
carries drc's ``(b, c, d, e)`` as ``(mu, s, upper, lower)`` with ``s > 0`` for the decreasing
``ryegrass`` response; the mapping was verified against drc's ``predict()`` inside
``generate.R``.

**A definitional mismatch, written here rather than absorbed into a tolerance.** drc's
``ED(..., interval = "delta")`` computes ``Estimate +/- qt(0.975, df) * SE`` on the original
concentration scale. robustmargin's delta interval is built on the log scale and exponentiated.
So the tests compare the point estimate, and the original-scale standard error reconstructed from
robustmargin's own endpoints (``sigma_log = ln(hi / lo) / (2 t)``, ``se = value * sigma_log``),
never the endpoints. The calibration is the t quantile on the residual df, not a normal one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.stats import t as student_t

from robustmargin import (
    Axis,
    ContinuousObservations,
    Definition,
    Fit,
    Status,
    Threshold,
    fit_dose_response,
    threshold,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tools" / "drc_reference"
_LEVEL = 0.95
_REL_TOL_PARAMS = 1e-4
_REL_TOL_COV = 1e-3
_REL_TOL_SE = 1e-3
_REL_TOL_BAND = 1e-3
_LINKS = [("logit", "LL4"), ("probit", "LN4")]


def _load() -> dict[str, Any]:
    with (_FIXTURE_DIR / "ryegrass_ll4.json").open() as fh:
        result: dict[str, Any] = json.load(fh)
    return result


def _fit(link: str) -> Fit:
    data = _load()["dataset"]
    obs = ContinuousObservations.from_observations(
        Axis(name="conc", unit="unit", scale="log"),
        severity=data["conc"],
        value=data["rootl"],
        outcome="response",
        direction="decreasing",
    )
    fit = fit_dose_response(obs, model="gaussian", link=link, upper="estimate", lower="estimate")
    assert fit.status is Status.OK
    return fit


@pytest.mark.parametrize(("link", "key"), _LINKS)
class TestRyegrassFit:
    def test_natural_parameters_match_mapped_drc_coefficients(self, link: str, key: str) -> None:
        expected = _load()[key]["mapped"]
        fit = _fit(link)
        assert fit.params is not None
        for name in ("mu", "s", "upper", "lower"):
            assert fit.params[name].value == pytest.approx(expected[name], rel=_REL_TOL_PARAMS)
            assert not fit.params[name].fixed

    def test_residual_matches_drc_rss_df_and_sd(self, link: str, key: str) -> None:
        ref = _load()[key]
        fit = _fit(link)
        assert fit.residual is not None
        assert fit.residual.df == ref["df_residual"]
        assert fit.residual.rss == pytest.approx(ref["rss"], rel=_REL_TOL_PARAMS)
        assert fit.residual.sigma == pytest.approx(ref["residual_sd"], rel=_REL_TOL_PARAMS)
        assert fit.residual.sigma == pytest.approx(ref["residual_sd_drc_summary"], rel=1e-3)

    def test_covariance_matches_delta_transformed_drc_vcov(self, link: str, key: str) -> None:
        """Covariance is the observed Hessian sigma-hat^2 (1/2 d2 RSS)^-1, sigma-hat^2 = RSS / df,
        compared with drc's vcov. The fixture's ``vcov_definitions`` block records whether drc's
        vcov equals this or nls's Gauss-Newton sigma-hat^2 (J'J)^-1.
        1e-3 relative, scaled by the pair's standard deviations: drc's vcov
        comes from a finite-difference Hessian in its own optimizer."""
        expected = _load()[key]["mapped_vcov"]
        names = expected["row_names"]
        assert names == ["mu", "s", "upper", "lower"]
        fit = _fit(link)
        assert fit.covariance is not None
        got = {n: i for i, n in enumerate(fit.covariance.names)}
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                want = expected["values"][i][j]
                scale = math.sqrt(expected["values"][i][i] * expected["values"][j][j])
                have = fit.covariance.matrix[got[a]][got[b]]
                assert have == pytest.approx(want, rel=_REL_TOL_COV, abs=_REL_TOL_COV * scale)

    def test_log_likelihood_is_the_gaussian_mle_value(self, link: str, key: str) -> None:
        ref = _load()[key]
        fit = _fit(link)
        assert fit.log_likelihood == pytest.approx(ref["logLik_gaussian_mle"], rel=_REL_TOL_PARAMS)

    def test_confidence_band_matches_drc_predict(self, link: str, key: str) -> None:
        """drc's ``predict(interval = "confidence")``: an identity-scale, t(df) band."""
        block = _load()[key]["predict_confidence"]
        fit = _fit(link)
        pred = fit.predict(block["conc"], level=block["level"])
        want = np.asarray(block["table"]["values"], dtype=float)
        assert np.allclose(pred.estimate, want[:, 0], rtol=_REL_TOL_PARAMS)
        assert np.allclose(pred.lo, want[:, 1], rtol=_REL_TOL_BAND)
        assert np.allclose(pred.hi, want[:, 2], rtol=_REL_TOL_BAND)


def _reconstructed_original_scale_se(t: Threshold, *, df: int) -> float:
    assert t.value is not None and t.lo is not None and t.hi is not None
    tq = float(student_t.ppf(0.5 + _LEVEL / 2.0, df))
    return t.value * math.log(t.hi / t.lo) / (2.0 * tq)


@pytest.mark.parametrize(("link", "key"), _LINKS)
class TestRyegrassEdDelta:
    def test_relative_ed_value_and_se_match_drc(self, link: str, key: str) -> None:
        block = _load()[key]["ED_relative"]
        fit = _fit(link)
        for p, row in zip(block["response_levels"], block["table"]["values"], strict=True):
            estimate, se, _lo, _hi = row
            t = threshold(
                fit,
                definition=Definition.relative(p / 100.0),
                interval_method="delta",
                level=_LEVEL,
            )
            assert t.status is Status.OK
            assert t.value == pytest.approx(estimate, rel=_REL_TOL_PARAMS)
            assert _reconstructed_original_scale_se(t, df=block["df_residual"]) == pytest.approx(
                se, rel=_REL_TOL_SE
            )

    def test_absolute_ed_value_matches_drc_and_se_matches_the_full_gradient_reference(
        self, link: str, key: str
    ) -> None:
        """The point estimate is drc's. The SE is held to the R-computed full-gradient delta
        over (b, c, d, e): drc's own absolute-ED SE holds the level's relative p fixed and is
        a different definition (pinned in the next test)."""
        data = _load()[key]
        block = data["ED_absolute"]
        full = data["ED_absolute_two_ways"]["full_gradient"]
        df = data["ED_absolute_two_ways"]["df_residual"]
        fit = _fit(link)
        levels = block["response_levels"]
        assert levels == data["ED_absolute_two_ways"]["response_levels"]
        for level, drc_row, full_row in zip(
            levels, block["table"]["values"], full["table"]["values"], strict=True
        ):
            t = threshold(
                fit,
                definition=Definition.absolute(level),
                interval_method="delta",
                level=_LEVEL,
            )
            assert t.status is Status.OK
            assert t.value == pytest.approx(drc_row[0], rel=_REL_TOL_PARAMS)
            assert t.value == pytest.approx(full_row[0], rel=_REL_TOL_PARAMS)
            assert _reconstructed_original_scale_se(t, df=df) == pytest.approx(
                full_row[1], rel=_REL_TOL_SE
            )

    def test_drc_absolute_ed_se_is_the_fixed_p_definition_not_the_full_gradient(
        self, link: str, key: str
    ) -> None:
        """Pins the definitional gap: drc's SE equals the fixture's fixed-p recomputation, and
        differs from the full-gradient SE (so the gap is not hidden inside a tolerance)."""
        data = _load()[key]
        drc_rows = data["ED_absolute"]["table"]["values"]
        two = data["ED_absolute_two_ways"]
        assert "fixed-p" in data["ED_absolute"]["definition"]
        for drc_row, fixed_row in zip(drc_rows, two["fixed_p"]["table"]["values"], strict=True):
            assert drc_row[1] == pytest.approx(fixed_row[1], rel=1e-8)
        gaps = [
            abs(d[1] - f[1]) / d[1]
            for d, f in zip(drc_rows, two["full_gradient"]["table"]["values"], strict=True)
        ]
        assert max(gaps) > 0.05

    def test_delta_interval_uses_the_t_quantile_not_the_normal_one(
        self, link: str, key: str
    ) -> None:
        """The log-scale half-width is ``t(df) * se_log``; a normal quantile would be narrower
        by a factor ``z / t(df)``."""
        block = _load()[key]["ED_relative"]
        fit = _fit(link)
        t = threshold(
            fit, definition=Definition.relative(0.5), interval_method="delta", level=_LEVEL
        )
        assert t.value is not None and t.lo is not None and t.hi is not None
        half_width = math.log(t.hi / t.lo) / 2.0
        tq = float(student_t.ppf(0.5 + _LEVEL / 2.0, block["df_residual"]))
        se_log = block["se_log"][1]  # ED50
        assert half_width == pytest.approx(tq * se_log, rel=_REL_TOL_SE)
