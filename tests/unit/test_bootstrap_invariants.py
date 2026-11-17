"""``Threshold`` invariants for bootstrap results.

* A bootstrap percentile interval may exclude its own point estimate, but only with a warning that
  says so. For the bootstrap methods an ``OK`` result with no censoring needs ``lo <= hi``, and
  ``lo > value`` or ``hi < value`` additionally needs a warning starting
  ``BOOTSTRAP_EXCLUDES_ESTIMATE:``; ``profile`` and ``delta`` keep ``lo <= value <= hi`` with no
  such escape. ``lo > hi`` is never accepted.
* When more than 10% of the replicates fail the bootstrap gives up: ``NOT_CONVERGED`` with
  ``interval_method="exact_bound"`` on a fit that is itself ``OK``. That is accepted exactly when a
  ``bootstrap`` summary is attached and its ``failed`` exceeds ``min(0.10, (1 - level) / 2)`` of
  ``n_resamples``.

Results are built directly from the deterministic fakes, so no resampling runs.
"""

from __future__ import annotations

import dataclasses

import pytest

from robustmargin import BootstrapSummary, Censoring, Status, Threshold
from robustmargin.testing import fake_fit, fake_threshold

_BOOTSTRAP_METHODS = ("parametric_bootstrap", "cell_bootstrap")
_SUMMARY = BootstrapSummary(
    n_resamples=100, failed=0, unreachable_low=0, unreachable_high=0, seed=1
)
_GIVE_UP = BootstrapSummary(
    n_resamples=100, failed=40, unreachable_low=0, unreachable_high=0, seed=1
)


_WARNING = (
    "BOOTSTRAP_EXCLUDES_ESTIMATE: the percentile interval excludes the estimate; z0 = -2.33",
)


def _bounded(model: str = "binomial") -> Threshold:
    t = fake_threshold(model=model)
    assert t.value is not None and t.lo is not None and t.hi is not None
    return t


@pytest.mark.parametrize("method", _BOOTSTRAP_METHODS)
class TestBootstrapIntervalMayExcludeTheEstimate:
    def test_a_lower_bound_above_the_value_is_accepted(self, method: str) -> None:
        t = _bounded()
        assert t.value is not None

        built = dataclasses.replace(
            t,
            interval_method=method,
            bootstrap=_SUMMARY,
            warnings=_WARNING,
            lo=t.value * 1.1,
            hi=t.value * 1.5,
        )

        assert built.lo is not None and built.lo > t.value

    def test_an_upper_bound_below_the_value_is_accepted(self, method: str) -> None:
        t = _bounded()
        assert t.value is not None

        built = dataclasses.replace(
            t,
            interval_method=method,
            bootstrap=_SUMMARY,
            warnings=_WARNING,
            lo=t.value * 0.5,
            hi=t.value * 0.9,
        )

        assert built.hi is not None and built.hi < t.value

    @pytest.mark.parametrize(("lo_scale", "hi_scale"), [(1.1, 1.5), (0.5, 0.9)])
    def test_an_interval_that_excludes_the_value_is_rejected_without_the_warning(
        self, method: str, lo_scale: float, hi_scale: float
    ) -> None:
        t = _bounded()
        assert t.value is not None

        with pytest.raises(ValueError):
            dataclasses.replace(
                t,
                interval_method=method,
                bootstrap=_SUMMARY,
                lo=t.value * lo_scale,
                hi=t.value * hi_scale,
            )

    def test_an_unrelated_warning_does_not_stand_in_for_the_excluded_estimate_warning(
        self, method: str
    ) -> None:
        t = _bounded()
        assert t.value is not None

        with pytest.raises(ValueError):
            dataclasses.replace(
                t,
                interval_method=method,
                bootstrap=_SUMMARY,
                warnings=("BOOTSTRAP_FAILURES: 3/100 bootstrap replicates failed",),
                lo=t.value * 1.1,
                hi=t.value * 1.5,
            )

    def test_an_interval_that_contains_the_value_needs_no_warning(self, method: str) -> None:
        t = _bounded()
        assert t.value is not None

        built = dataclasses.replace(
            t,
            interval_method=method,
            bootstrap=_SUMMARY,
            lo=t.value * 0.9,
            hi=t.value * 1.1,
        )

        assert built.warnings == t.warnings

    def test_a_lower_bound_above_the_upper_bound_is_rejected(self, method: str) -> None:
        t = _bounded()
        assert t.value is not None

        with pytest.raises(ValueError):
            dataclasses.replace(
                t,
                interval_method=method,
                bootstrap=_SUMMARY,
                warnings=_WARNING,
                lo=t.value * 1.5,
                hi=t.value * 1.1,
            )


@pytest.mark.parametrize("method", ["profile", "delta"])
class TestModelIntervalsStillContainTheEstimate:
    def test_a_lower_bound_above_the_value_is_rejected(self, method: str) -> None:
        t = _bounded()
        assert t.value is not None

        with pytest.raises(ValueError):
            dataclasses.replace(t, interval_method=method, lo=t.value * 1.1, hi=t.value * 1.5)

    def test_an_upper_bound_below_the_value_is_rejected(self, method: str) -> None:
        t = _bounded()
        assert t.value is not None

        with pytest.raises(ValueError):
            dataclasses.replace(t, interval_method=method, lo=t.value * 0.5, hi=t.value * 0.9)

    def test_the_excluded_estimate_warning_does_not_loosen_a_model_interval(
        self, method: str
    ) -> None:
        t = _bounded()
        assert t.value is not None

        with pytest.raises(ValueError):
            dataclasses.replace(
                t,
                interval_method=method,
                warnings=_WARNING,
                lo=t.value * 1.1,
                hi=t.value * 1.5,
            )


def _give_up(model: str, bootstrap: BootstrapSummary | None, *, level: float = 0.95) -> Threshold:
    t = fake_threshold(model=model)
    assert t.fit.status is Status.OK
    return dataclasses.replace(
        t,
        level=level,
        status=Status.NOT_CONVERGED,
        censoring=Censoring.NONE,
        interval_method="exact_bound",
        value=None,
        lo=None,
        hi=None,
        bootstrap=bootstrap,
    )


@pytest.mark.parametrize("model", ["binomial", "gaussian"])
class TestGiveUpOnAnOkFit:
    def test_not_converged_on_an_ok_fit_is_accepted_with_a_bootstrap_summary(
        self, model: str
    ) -> None:
        t = _give_up(model, _GIVE_UP)

        assert t.status is Status.NOT_CONVERGED and t.fit.status is Status.OK
        assert t.bootstrap == _GIVE_UP

    @pytest.mark.parametrize(
        ("level", "failed", "accepted"),
        [
            # Cut-off min(0.10, (1 - level) / 2) * n_resamples, with n_resamples = 100.
            (0.95, 2, False),
            (0.95, 3, True),
            (0.99, 0, False),
            (0.99, 1, True),
            (0.7, 10, False),
            (0.7, 11, True),
        ],
    )
    def test_the_give_up_needs_more_failures_than_the_levels_cut_off(
        self, model: str, level: float, failed: int, accepted: bool
    ) -> None:
        summary = BootstrapSummary(
            n_resamples=100, failed=failed, unreachable_low=0, unreachable_high=0, seed=1
        )

        if accepted:
            assert _give_up(model, summary, level=level).bootstrap == summary
        else:
            with pytest.raises(ValueError):
                _give_up(model, summary, level=level)

    def test_not_converged_on_an_ok_fit_is_rejected_without_a_bootstrap_summary(
        self, model: str
    ) -> None:
        with pytest.raises(ValueError):
            _give_up(model, None)


class TestExcludedEstimateWarning:
    def test_threshold_warns_and_reports_the_percentile_interval_unclamped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import math

        import numpy as np
        from scipy.stats import norm

        from robustmargin import Axis, Definition, Observations, fit_dose_response, threshold
        from robustmargin.intervals import _BootstrapDesign

        x = np.array([0.0, *np.logspace(-1.0, 1.0, 10)])
        with np.errstate(divide="ignore"):
            p = np.where(x == 0.0, 1.0, 1.0 - norm.cdf((np.log(x) - math.log(2.5)) / 0.5))
        obs = Observations.from_counts(
            Axis(name="severity", unit="unit", scale="log"),
            severity=x.tolist(),
            successes=np.random.default_rng(20261114).binomial(400, p).tolist(),
            trials=[400] * x.size,
            outcome="success",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
        point = threshold(
            fit,
            definition=Definition.absolute(0.5),
            interval_method="delta",
            level=0.95,
        )
        assert point.value is not None
        # Every replicate lands well above the estimate, as a biased estimator would give.
        thetas = math.log(point.value) + 0.5 + np.linspace(0.0, 0.1, 100)
        monkeypatch.setattr(
            _BootstrapDesign, "theta", lambda self, first, second, index: float(thetas[index])
        )

        t = threshold(
            fit,
            definition=Definition.absolute(0.5),
            interval_method="parametric_bootstrap",
            level=0.95,
            n_resamples=100,
            seed=1,
        )

        assert any(w.startswith("BOOTSTRAP_EXCLUDES_ESTIMATE:") for w in t.warnings)
        assert t.status is Status.OK and t.censoring is Censoring.NONE
        assert t.value == point.value
        assert t.lo == pytest.approx(math.exp(np.quantile(thetas, 0.025, method="weibull")))
        assert t.hi == pytest.approx(math.exp(np.quantile(thetas, 0.975, method="weibull")))
        assert t.lo is not None and t.lo > t.value


class TestOldTagsRefuseNewContent:
    """A result tagged ``"1"`` or ``"2"`` cannot hold what only ``"3"`` can express."""

    @pytest.mark.parametrize("version", ["1", "2"])
    def test_an_isotonic_fit_with_an_old_tag_is_rejected(self, version: str) -> None:
        with pytest.raises(ValueError):
            dataclasses.replace(fake_fit(model="isotonic"), schema_version=version)

    @pytest.mark.parametrize("version", ["1", "2"])
    def test_a_parametric_fit_with_an_old_tag_is_still_accepted(self, version: str) -> None:
        assert dataclasses.replace(fake_fit(), schema_version=version).schema_version == version

    @pytest.mark.parametrize("version", ["1", "2"])
    @pytest.mark.parametrize("method", _BOOTSTRAP_METHODS)
    def test_a_bootstrap_method_with_an_old_tag_is_rejected(
        self, version: str, method: str
    ) -> None:
        with pytest.raises(ValueError):
            dataclasses.replace(fake_threshold(), schema_version=version, interval_method=method)

    @pytest.mark.parametrize("version", ["1", "2"])
    def test_a_bootstrap_summary_with_an_old_tag_is_rejected(self, version: str) -> None:
        with pytest.raises(ValueError):
            dataclasses.replace(fake_threshold(), schema_version=version, bootstrap=_SUMMARY)

    @pytest.mark.parametrize("version", ["1", "2"])
    def test_an_isotonic_threshold_with_an_old_tag_is_rejected(self, version: str) -> None:
        with pytest.raises(ValueError):
            dataclasses.replace(fake_threshold(model="isotonic"), schema_version=version)

    @pytest.mark.parametrize("version", ["1", "2"])
    @pytest.mark.parametrize("method", ["profile", "delta"])
    def test_a_model_interval_threshold_with_an_old_tag_is_still_accepted(
        self, version: str, method: str
    ) -> None:
        t = dataclasses.replace(fake_threshold(interval_method=method), schema_version=version)

        assert t.schema_version == version
