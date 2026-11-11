"""Ratio of two continuous thresholds against R computed from two separate drc fits.

**Reads only the committed ``tools/drc_reference/salba_separate.json``. Never runs R.**

Each herbicide is fitted with its own ``LL.4`` (its own residual variance and df), and the ED50
ratio is computed in R from the two fits. The exact formulas are written into the fixture
(``log_delta_formula``, ``fieller_formula``) and in ``tools/drc_reference/README.md``:

- log-delta: ``nu = (La + Lb)^2 / (La^2/df_a + Lb^2/df_b)`` on the log-scale variances, interval
  ``rho * exp(+-qt(0.975, nu) * sqrt(La + Lb))``;
- Fieller (independent, ``C = 0``): ``nu = (Va + rho^2 Vb)^2 / (Va^2/df_a + rho^4 Vb^2/df_b)``
  at ``rho = ED_a / ED_b``, ``t = qt(0.975, nu)``, roots of
  ``(ED_b^2 - t^2 Vb) r^2 - 2 ED_a ED_b r + (ED_a^2 - t^2 Va) = 0``.

drc's pooled-sigma ``EDcomp`` output (``info_only_pooled_sigma``) is not a reference; only the
point estimate of its tight joint refit is checked, because least squares separates by curve when
every parameter is curve-specific.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from scipy.stats import t as student_t

from robustmargin import (
    Axis,
    ContinuousObservations,
    Definition,
    Fit,
    IntervalShape,
    Status,
    Threshold,
    fit_dose_response,
    ratio_interval,
    threshold,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tools" / "drc_reference"
_LEVEL = 0.95
_REL_TOL_PARAMS = 1e-4
_REL_TOL_COV = 1e-3
_REL_TOL_SE = 1e-3
_REL_TOL_ENDPOINTS = 1e-3


def _load() -> dict[str, Any]:
    with (_FIXTURE_DIR / "salba_separate.json").open() as fh:
        result: dict[str, Any] = json.load(fh)
    return result


def _fit(herbicide: str) -> Fit:
    block = _load()["separate_fits"][herbicide]
    obs = ContinuousObservations.from_observations(
        Axis(name="dose", unit="unit", scale="log"),
        severity=block["dataset"]["dose"],
        value=block["dataset"]["dry_matter"],
        outcome="response",
        direction="decreasing",
    )
    fit = fit_dose_response(obs, model="gaussian", link="logit", upper="estimate", lower="estimate")
    assert fit.status is Status.OK
    return fit


def _ed50(herbicide: str) -> Threshold:
    return threshold(
        _fit(herbicide),
        definition=Definition.relative(0.5),
        interval_method="delta",
        level=_LEVEL,
    )


def _pair() -> tuple[Threshold, Threshold]:
    ref = _load()
    return _ed50(ref["numerator"]), _ed50(ref["denominator"])


@pytest.mark.parametrize("which", ["numerator", "denominator"])
class TestSeparateFits:
    def test_parameters_and_residual_match_drc(self, which: str) -> None:
        ref = _load()
        block = ref["separate_fits"][ref[which]]
        fit = _fit(ref[which])
        assert fit.params is not None and fit.residual is not None
        for name in ("mu", "s", "upper", "lower"):
            assert fit.params[name].value == pytest.approx(
                block["mapped"][name], rel=_REL_TOL_PARAMS
            )
        assert fit.residual.df == block["df_residual"]
        assert fit.residual.rss == pytest.approx(block["rss"], rel=_REL_TOL_PARAMS)
        assert fit.residual.sigma == pytest.approx(block["residual_sd"], rel=_REL_TOL_PARAMS)

    def test_covariance_matches_delta_transformed_drc_vcov(self, which: str) -> None:
        ref = _load()
        expected = ref["separate_fits"][ref[which]]["mapped_vcov"]
        fit = _fit(ref[which])
        assert fit.covariance is not None
        idx = {n: i for i, n in enumerate(fit.covariance.names)}
        names = expected["row_names"]
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                want = expected["values"][i][j]
                scale = math.sqrt(expected["values"][i][i] * expected["values"][j][j])
                have = fit.covariance.matrix[idx[a]][idx[b]]
                assert have == pytest.approx(want, rel=_REL_TOL_COV, abs=_REL_TOL_COV * scale)

    def test_ed50_value_and_log_scale_se_match_drc(self, which: str) -> None:
        ref = _load()
        block = ref["separate_fits"][ref[which]]
        t = _ed50(ref[which])
        assert t.value is not None and t.lo is not None and t.hi is not None
        assert t.value == pytest.approx(block["ED50"]["estimate"], rel=_REL_TOL_PARAMS)
        tq = float(student_t.ppf(0.5 + _LEVEL / 2.0, block["df_residual"]))
        sigma_log = math.log(t.hi / t.lo) / (2.0 * tq)
        assert sigma_log == pytest.approx(block["ED50"]["se_log"], rel=_REL_TOL_SE)


class TestRatioFromSeparateFits:
    def test_estimate_matches_the_ratio_of_drc_ed50s(self) -> None:
        ref = _load()
        want = ref["ratio_ED50_from_separate_fits"]["log_delta"]["estimate"]
        a, b = _pair()
        result = ratio_interval(a, b, method="log_delta", dependence="independent", level=_LEVEL)
        assert result.estimate == pytest.approx(want, rel=_REL_TOL_PARAMS)

    def test_estimate_matches_the_pooled_edcomp_point_estimate(self) -> None:
        """With every parameter curve-specific the pooled fit's ED50s are the separate fits'
        (the intervals are not, because sigma is pooled). The reference is drc's tight joint
        refit: its default joint fit stops short of the minimum."""
        ref = _load()
        pooled = ref["info_only_pooled_sigma"]
        assert pooled["label"] == "info_only_pooled_sigma"
        assert pooled["curve_order"] == ref["herbicide_levels"]
        assert pooled["rss_tight_fit"] <= pooled["rss_default_fit"]
        a, b = _pair()
        result = ratio_interval(a, b, method="fieller", dependence="independent", level=_LEVEL)
        assert result.estimate == pytest.approx(
            pooled["tight_refit"]["EDcomp_delta"]["values"][0][0], rel=_REL_TOL_PARAMS
        )

    def test_log_delta_interval_uses_welch_satterthwaite_t(self) -> None:
        want = _load()["ratio_ED50_from_separate_fits"]["log_delta"]
        a, b = _pair()
        result = ratio_interval(a, b, method="log_delta", dependence="independent", level=_LEVEL)
        assert result.shape is IntervalShape.BOUNDED
        assert result.lo == pytest.approx(want["lo"], rel=_REL_TOL_ENDPOINTS)
        assert result.hi == pytest.approx(want["hi"], rel=_REL_TOL_ENDPOINTS)

    def test_fieller_interval_uses_welch_satterthwaite_t_at_the_point_estimate(self) -> None:
        want = _load()["ratio_ED50_from_separate_fits"]["fieller"]
        a, b = _pair()
        result = ratio_interval(a, b, method="fieller", dependence="independent", level=_LEVEL)
        assert result.shape is IntervalShape.BOUNDED
        assert result.lo == pytest.approx(want["lo"], rel=_REL_TOL_ENDPOINTS)
        assert result.hi == pytest.approx(want["hi"], rel=_REL_TOL_ENDPOINTS)

    def test_welch_t_is_wider_than_a_normal_quantile_would_give(self) -> None:
        """Guards against a z quantile being used for a continuous ratio: the fixture's
        log-delta t quantile exceeds the normal one, so the interval must exceed the z one."""
        want = _load()["ratio_ED50_from_separate_fits"]["log_delta"]
        a, b = _pair()
        result = ratio_interval(a, b, method="log_delta", dependence="independent", level=_LEVEL)
        assert result.lo is not None and result.hi is not None
        z = 1.959963984540054
        assert want["t_quantile"] > z
        half = math.log(result.hi / result.lo) / 2.0
        assert half == pytest.approx(want["t_quantile"] * want["sigma_log"], rel=_REL_TOL_ENDPOINTS)
        assert half > z * want["sigma_log"]
