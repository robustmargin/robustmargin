"""Reference-value parity for :func:`~robustmargin.diagnose` against R's ``glm`` on ``finney71``.

This file never runs R and never needs R installed -- it reads only the committed
``tools/drc_reference/finney71.json``. Every one of AIC, deviance and Pearson chi-square is
**derived from R's own fitted coefficients and log-likelihood** (``glm.<link>.coef``,
``glm.<link>.logLik``): R's fitted probability at each dose is
``linkinv(intercept + slope * log(dose))`` for the ``cbind(affected, total - affected) ~
log(dose)`` model the fixture's own ``glm`` block describes, and deviance/Pearson are the
ordinary binomial GOF formulas applied to that fitted probability. This is the same "recompute
from R's coefficients" approach ``tools/drc_reference/generate.R``'s own ``observed_hessian_check``
uses for covariance.

``generate.R`` also has R itself compute and emit ``deviance``, ``pearson_chi2``, ``df_residual``
and ``AIC`` per ``glm`` link directly, as a second, R-native cross-check
(``TestAgainstRNativeGofFields`` below). A missing fixture field fails those tests with a
``KeyError`` naming it: that means the fixture needs regenerating with the pinned R container
(``tools/drc_reference/README.md``), not that the formula is wrong -- and a reference-value
drift is a finding, not a test bug.

**Factual note:** R's ``glm`` does not "drop" the exact ``dose == 0`` row --
``tools/drc_reference/generate.R`` itself subsets that row *out of the data* before ever calling
``glm``, because ``log(0)`` is undefined and would make the ``glm`` call fail outright, not
because ``glm`` has any special handling for an all-success control. The resulting 5-row ``glm``
fit's ``df.residual`` (5 rows minus 2 estimated parameters) is what ``df == 3`` is compared
against here. This is a **different** exclusion from robustmargin's own: ``diagnose()`` excludes
the ``dose == 0`` cell *structurally* -- a log-axis cell at severity exactly ``0`` when ``upper``
is fixed at exactly ``1.0`` -- because that cell has zero variance and zero deviance by
construction under robustmargin's own curve, not because a numerically-saturated fitted
probability was excluded (excluding by *computed value* rather than by structure would make
``df`` depend on the link). The two exclusions happen to agree on `finney71` (both drop exactly
the ``dose == 0`` row), which is what makes the comparison below meaningful at all. drc's own
`drm`, by contrast, fits the *full* 6-row `finney71` (its `LN.2`/`LL.2` forms handle
`dose == 0` on the log scale directly), so drc's residual `df` is 4, not 3 -- recorded next to
the fixture by ``generate.R``, not reproduced here since nothing in this file compares against
drc's own fit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.special import xlogy
from scipy.stats import norm

from robustmargin import Axis, Observations, Status, diagnose, fit_dose_response

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tools" / "drc_reference"
_REL_TOL = 1e-4


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


def _linkinv(eta: np.ndarray, link: str) -> np.ndarray:
    if link == "probit":
        return norm.cdf(eta)
    if link == "logit":
        return 1.0 / (1.0 + np.exp(-eta))
    return -np.expm1(-np.exp(eta))


def _r_gof(link: str) -> dict[str, float]:
    """Deviance, Pearson chi-square, df and AIC for R's own ``glm(link)`` fit on the ``dose >
    0`` subset, computed directly from ``coef``/``logLik`` already in the fixture."""
    fixture = _load("finney71.json")
    dose = np.array(fixture["dataset"]["dose"], dtype=float)
    total = np.array(fixture["dataset"]["total"], dtype=float)
    affected = np.array(fixture["dataset"]["affected"], dtype=float)
    mask = dose > 0.0
    dose, total, affected = dose[mask], total[mask], affected[mask]
    log_dose = np.log(dose)

    coef = fixture["glm"][link]["coef"]
    eta = coef["(Intercept)"] + coef["log(dose)"] * log_dose
    p_fail = _linkinv(eta, link)

    mu = total * p_fail
    failed = total - affected
    dev_terms = xlogy(affected, affected / mu) + xlogy(failed, failed / (total - mu))
    deviance = float(2.0 * np.sum(dev_terms))
    pearson = float(np.sum((affected - mu) ** 2 / (total * p_fail * (1.0 - p_fail))))
    df = int(len(dose) - 2)
    log_lik = float(fixture["glm"][link]["logLik"])
    aic = -2.0 * log_lik + 2.0 * 2

    return {"deviance": deviance, "pearson_chi2": pearson, "df": df, "aic": aic}


@pytest.fixture(scope="module")
def _fits() -> dict[str, Any]:
    obs = _finney71_observations()
    fits = {}
    for link in ("probit", "logit", "cloglog"):
        fit = fit_dose_response(obs, model="binomial", link=link, upper=1.0, lower=0.0)
        assert fit.status is Status.OK, f"finney71 {link} fit must converge (test_models_drc.py)"
        fits[link] = fit
    return fits


class TestGoodnessOfFitAgainstRsOwnCoefficients:
    @pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
    def test_deviance_matches_r_within_tolerance(self, _fits: dict[str, Any], link: str) -> None:
        expected = _r_gof(link)
        diag = diagnose(_fits[link])

        assert diag.deviance == pytest.approx(expected["deviance"], rel=_REL_TOL)

    @pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
    def test_pearson_chi2_matches_r_within_tolerance(
        self, _fits: dict[str, Any], link: str
    ) -> None:
        expected = _r_gof(link)
        diag = diagnose(_fits[link])

        assert diag.pearson_chi2 == pytest.approx(expected["pearson_chi2"], rel=_REL_TOL)

    @pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
    def test_df_excludes_the_all_success_zero_dose_control(
        self, _fits: dict[str, Any], link: str
    ) -> None:
        """5 dose>0 rows minus 2 estimated parameters (mu, s; upper/lower fixed). robustmargin
        excludes the dose=0 control *structurally* (a log-axis severity-0 cell under a fixed
        upper=1.0) -- not because its fitted probability happens to round to exactly 1.0. R's
        glm never even sees that row: generate.R subsets it out of the data before fitting,
        since log(0) is undefined (module docstring's factual note). The two are different
        exclusions that happen to remove the same row here."""
        expected = _r_gof(link)
        diag = diagnose(_fits[link])

        assert diag.df == expected["df"] == 3


class TestLinkComparisonAicAgainstR:
    def test_link_aic_matches_r_for_every_link(self, _fits: dict[str, Any]) -> None:
        diag = diagnose(_fits["probit"])

        for link in ("probit", "logit", "cloglog"):
            expected_aic = _r_gof(link)["aic"]
            assert diag.link_aic[link] == pytest.approx(expected_aic, rel=_REL_TOL)
            assert diag.link_status[link] is Status.OK


class TestAgainstRNativeGofFields:
    """``finney71.json``'s ``glm.<link>`` blocks carry R's *own*
    ``deviance()``/``sum(residuals(g, type="pearson")^2)``/``df.residual()``/``AIC()`` directly,
    written by ``tools/drc_reference/generate.R``. These tests are a second, fully R-native
    cross-check, independent of this file's own Python recomputation above. A missing field
    fails with a clear ``KeyError`` -- that failure means "regenerate the fixture", not a
    statistical finding.
    """

    @pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
    def test_r_native_deviance_matches_diagnose(self, _fits: dict[str, Any], link: str) -> None:
        fixture = _load("finney71.json")
        r_native_deviance = fixture["glm"][link]["deviance"]
        diag = diagnose(_fits[link])

        assert diag.deviance == pytest.approx(r_native_deviance, rel=_REL_TOL)

    @pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
    def test_r_native_pearson_chi2_matches_diagnose(self, _fits: dict[str, Any], link: str) -> None:
        fixture = _load("finney71.json")
        r_native_pearson = fixture["glm"][link]["pearson_chi2"]
        diag = diagnose(_fits[link])

        assert diag.pearson_chi2 == pytest.approx(r_native_pearson, rel=_REL_TOL)

    @pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
    def test_r_native_aic_matches_diagnose(self, _fits: dict[str, Any], link: str) -> None:
        fixture = _load("finney71.json")
        r_native_aic = fixture["glm"][link]["AIC"]
        diag = diagnose(_fits[link])

        assert diag.link_aic[link] == pytest.approx(r_native_aic, rel=_REL_TOL)


class TestDrcVsGlmDfComparisonNote:
    """The factual note recorded next to the fixture by ``generate.R``: drc's ``drm()`` fits the
    full 6-row ``finney71`` (its own ``df.residual`` is 4), where ``glm()`` here and
    robustmargin's own ``diagnose()`` both report 3 -- for two different reasons that happen to
    agree, not because drc "gets it wrong". This class needs the ``df_comparison_note`` field
    ``generate.R`` emits; a missing field is a ``KeyError``, same treatment as
    ``TestAgainstRNativeGofFields`` above.
    """

    def test_drc_df_residual_is_4(self) -> None:
        fixture = _load("finney71.json")
        note = fixture["df_comparison_note"]

        assert note["drc_df_residual"] == 4

    def test_glm_df_residual_is_3(self) -> None:
        fixture = _load("finney71.json")
        note = fixture["df_comparison_note"]

        assert note["glm_df_residual"] == 3

    def test_robustmargin_diagnose_df_matches_glm_not_drc(self, _fits: dict[str, Any]) -> None:
        fixture = _load("finney71.json")
        note = fixture["df_comparison_note"]

        diag = diagnose(_fits["probit"])

        assert diag.df == note["glm_df_residual"] == 3
        assert diag.df != note["drc_df_residual"]


class TestSparseCellsWarningDoesNotFireOnFinney71:
    """Every counted finney71 cell has an expected count
    well above 1 (verified directly against the same R-derived fitted probabilities the GOF
    tests above use, not just asserted) -- so ``SPARSE_CELLS: `` must not fire here."""

    @pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
    def test_every_counted_cell_has_expected_count_above_1(self, link: str) -> None:
        fixture = _load("finney71.json")
        dose = np.array(fixture["dataset"]["dose"], dtype=float)
        total = np.array(fixture["dataset"]["total"], dtype=float)
        mask = dose > 0.0
        dose, total = dose[mask], total[mask]
        log_dose = np.log(dose)
        coef = fixture["glm"][link]["coef"]
        eta = coef["(Intercept)"] + coef["log(dose)"] * log_dose
        p_fail = _linkinv(eta, link)
        p_success = 1.0 - p_fail

        assert np.all(total * p_success >= 1.0)
        assert np.all(total * (1.0 - p_success) >= 1.0)

    @pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
    def test_sparse_cells_warning_does_not_fire(self, _fits: dict[str, Any], link: str) -> None:
        diag = diagnose(_fits[link])

        assert not any(w.startswith("SPARSE_CELLS: ") for w in diag.warnings)
