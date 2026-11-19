"""Argument-combination guards for clustered and paired inference.

For every guard the *valid* neighbour of the call is exercised too, so a guard test cannot pass
merely because the feature does not exist yet. Messages are matched only where the behaviour
specification quotes them.
"""

from __future__ import annotations

import pytest
from _cluster_fixtures import (
    AXIS,
    clustered_counts,
    clustered_fit,
    clustered_threshold,
    paired_fits,
)

from robustmargin import (
    Censoring,
    ContinuousObservations,
    Definition,
    Fit,
    Observations,
    Status,
    Threshold,
    check_monotonicity,
    diagnose,
    fit_dose_response,
    ratio_interval,
    threshold,
)
from robustmargin.testing import fake_fit


def _fit(obs: Observations, *, dependence: str | None = None) -> Fit:
    return fit_dose_response(
        obs, model="binomial", link="probit", upper=1.0, lower=0.0, dependence=dependence
    )


def _unclustered() -> Observations:
    return Observations.from_counts(
        AXIS,
        severity=[0.0, 1.0, 2.0, 4.0, 8.0],
        successes=[40, 36, 28, 14, 4],
        trials=[40] * 5,
        outcome="success",
        direction="decreasing",
    )


def _continuous_clustered() -> ContinuousObservations:
    severity = [0.0, 1.0, 2.0, 4.0] * 3
    return ContinuousObservations.from_observations(
        AXIS,
        severity=severity,
        value=[10.0, 8.0, 5.0, 2.0, 9.5, 8.5, 4.0, 2.5, 10.5, 7.5, 5.5, 1.5],
        outcome="response",
        direction="decreasing",
        cluster=["a"] * 4 + ["b"] * 4 + ["c"] * 4,
    )


class TestFitDoseResponseGuards:
    def test_clustered_dependence_fits_clustered_binomial_input(self) -> None:
        fit = _fit(clustered_counts(), dependence="clustered")

        assert fit.dependence == "clustered"

    def test_clustered_dependence_requires_clustered_observations(self) -> None:
        assert clustered_fit().dependence == "clustered"  # the valid neighbour exists
        with pytest.raises(ValueError):
            _fit(_unclustered(), dependence="clustered")

    def test_clustered_dependence_on_gaussian_raises_naming_the_missing_feature(self) -> None:
        with pytest.raises(ValueError, match="clustered continuous inference is not implemented"):
            fit_dose_response(
                _continuous_clustered(),
                model="gaussian",
                link="logit",
                upper=10.0,
                lower=1.0,
                dependence="clustered",
            )

    def test_clustered_dependence_on_isotonic_raises(self) -> None:
        with pytest.raises(ValueError, match="clustered continuous inference is not implemented"):
            fit_dose_response(clustered_counts(), model="isotonic", dependence="clustered")

    @pytest.mark.parametrize("dependence", ["paired", "bogus", ""])
    def test_unknown_dependence_raises(self, dependence: str) -> None:
        assert clustered_fit().dependence == "clustered"
        with pytest.raises(ValueError):
            _fit(clustered_counts(), dependence=dependence)

    def test_omitting_dependence_on_clustered_input_still_raises(self) -> None:
        # The clustered fit must be reachable first, or this would pass without the feature.
        assert clustered_fit().dependence == "clustered"
        with pytest.raises(ValueError):
            _fit(clustered_counts())

    def test_unclustered_input_defaults_to_independent_and_stores_no_cluster_cells(self) -> None:
        fit = _fit(_unclustered())

        assert fit.dependence == "independent"
        assert fit.cluster_cells is None

    def test_two_clusters_at_fixed_asymptotes_raise_because_g_does_not_exceed_k(self) -> None:
        assert clustered_fit(3).dependence == "clustered"
        with pytest.raises(ValueError, match="too few clusters"):
            _fit(clustered_counts(2), dependence="clustered")

    def test_four_clusters_with_estimated_asymptotes_raise_because_g_does_not_exceed_k(
        self,
    ) -> None:
        assert clustered_fit(4).dependence == "clustered"
        with pytest.raises(ValueError, match="too few clusters"):
            fit_dose_response(
                clustered_counts(4),
                model="binomial",
                link="probit",
                upper="estimate",
                lower="estimate",
                dependence="clustered",
            )

    def test_a_clustered_fit_carries_a_few_clusters_warning_below_k(self) -> None:
        fit = clustered_fit(4)

        assert any(w.startswith("FEW_CLUSTERS: ") for w in fit.warnings)

    def test_clustered_point_estimate_and_log_likelihood_equal_the_independent_fit(self) -> None:
        clustered = clustered_fit()
        independent = clustered_fit(dependence="independent")

        assert clustered.status is independent.status
        assert clustered.log_likelihood == independent.log_likelihood
        assert clustered.params == independent.params

    def test_clustered_covariance_differs_from_the_model_based_one(self) -> None:
        clustered = clustered_fit()
        independent = clustered_fit(dependence="independent")

        assert clustered.covariance is not None and independent.covariance is not None
        assert clustered.covariance.matrix != independent.covariance.matrix

    def test_clustered_fit_attaches_a_cluster_sign_test_monotonicity_check(self) -> None:
        fit = clustered_fit()

        assert fit.monotonicity is not None
        assert fit.monotonicity.dependence == "clustered"
        assert fit.monotonicity.method == "cluster_sign_test_one_sided_holm"

    def test_independent_fit_of_clustered_data_keeps_the_fisher_check(self) -> None:
        fit = clustered_fit(dependence="independent")

        assert fit.monotonicity is not None
        assert fit.monotonicity.dependence == "independent"
        assert fit.monotonicity.method == "fisher_exact_one_sided_holm"


class TestCheckMonotonicityGuards:
    def test_clustered_dependence_returns_the_cluster_sign_test(self) -> None:
        from robustmargin import ClusterAdjacentPair

        check = check_monotonicity(clustered_counts(), dependence="clustered")

        assert check.dependence == "clustered"
        assert check.method == "cluster_sign_test_one_sided_holm"
        assert len(check.pairs) == 5
        assert all(isinstance(p, ClusterAdjacentPair) for p in check.pairs)

    def test_holm_adjusted_p_is_never_below_the_raw_p(self) -> None:
        check = check_monotonicity(clustered_counts(), dependence="clustered")

        assert all(p.holm_adjusted_p >= p.p_value for p in check.pairs)

    def test_clustered_dependence_requires_clustered_observations(self) -> None:
        assert check_monotonicity(clustered_counts(), dependence="clustered").pairs
        with pytest.raises(ValueError):
            check_monotonicity(_unclustered(), dependence="clustered")

    def test_clustered_dependence_on_continuous_observations_raises(self) -> None:
        assert check_monotonicity(clustered_counts(), dependence="clustered").pairs
        with pytest.raises(ValueError):
            check_monotonicity(_continuous_clustered(), dependence="clustered")

    def test_independent_dependence_on_clustered_input_is_still_the_fisher_check(self) -> None:
        check = check_monotonicity(clustered_counts(), dependence="independent")

        assert check.method == "fisher_exact_one_sided_holm"

    def test_omitting_dependence_on_clustered_input_still_raises(self) -> None:
        assert check_monotonicity(clustered_counts(), dependence="clustered").pairs
        with pytest.raises(ValueError):
            check_monotonicity(clustered_counts())

    def test_unknown_dependence_raises(self) -> None:
        assert check_monotonicity(clustered_counts(), dependence="clustered").pairs
        with pytest.raises(ValueError):
            check_monotonicity(clustered_counts(), dependence="paired")


class TestThresholdGuards:
    def test_clustered_fit_requires_dependence_clustered(self) -> None:
        fit = clustered_fit()
        for dependence in (None, "independent"):
            with pytest.raises(ValueError):
                threshold(
                    fit,
                    definition=Definition.absolute(0.5),
                    interval_method="delta",
                    level=0.95,
                    dependence=dependence,
                )

    def test_clustered_dependence_requires_a_clustered_fit(self) -> None:
        assert clustered_threshold().dependence == "clustered"
        independent = clustered_fit(dependence="independent")

        with pytest.raises(ValueError):
            threshold(
                independent,
                definition=Definition.absolute(0.5),
                interval_method="delta",
                level=0.95,
                dependence="clustered",
            )

    @pytest.mark.parametrize("method", ["profile", "parametric_bootstrap", "cell_bootstrap"])
    def test_independence_interval_methods_raise_on_a_clustered_fit(self, method: str) -> None:
        assert clustered_threshold().dependence == "clustered"

        with pytest.raises(ValueError):
            threshold(
                clustered_fit(),
                definition=Definition.absolute(0.5),
                interval_method=method,
                level=0.95,
                dependence="clustered",
            )

    def test_cluster_bootstrap_on_an_independent_fit_raises(self) -> None:
        assert clustered_threshold().dependence == "clustered"

        with pytest.raises(ValueError):
            threshold(
                clustered_fit(dependence="independent"),
                definition=Definition.absolute(0.5),
                interval_method="cluster_bootstrap",
                level=0.95,
                dependence="independent",
                n_resamples=19,
                seed=1,
            )

    def test_cluster_bootstrap_on_an_unclustered_fit_raises(self) -> None:
        assert clustered_threshold().dependence == "clustered"

        with pytest.raises(ValueError):
            threshold(
                _fit(_unclustered()),
                definition=Definition.absolute(0.5),
                interval_method="cluster_bootstrap",
                level=0.95,
                n_resamples=19,
                seed=1,
            )

    def test_cluster_bootstrap_on_a_gaussian_fit_raises(self) -> None:
        assert clustered_threshold().dependence == "clustered"

        with pytest.raises(ValueError):
            threshold(
                fake_fit(model="gaussian"),
                definition=Definition.absolute(5.0),
                interval_method="cluster_bootstrap",
                level=0.95,
                dependence="clustered",
                n_resamples=19,
                seed=1,
            )

    def test_resample_arguments_are_refused_with_the_delta_method(self) -> None:
        assert clustered_threshold().dependence == "clustered"

        for kwargs in ({"n_resamples": 19}, {"seed": 1}):
            with pytest.raises(ValueError):
                clustered_threshold(interval_method="delta", **kwargs)  # type: ignore[arg-type]

    def test_cluster_bootstrap_is_a_bootstrap_method_and_records_its_summary(self) -> None:
        t = clustered_threshold(interval_method="cluster_bootstrap", n_resamples=19, seed=5)

        assert t.interval_method == "cluster_bootstrap"
        assert t.dependence == "clustered"
        assert t.bootstrap is not None
        assert t.bootstrap.n_resamples == 19
        assert t.bootstrap.seed == 5

    def test_the_same_seed_reproduces_the_cluster_bootstrap_result(self) -> None:
        first = clustered_threshold(interval_method="cluster_bootstrap", n_resamples=19, seed=5)
        second = clustered_threshold(interval_method="cluster_bootstrap", n_resamples=19, seed=5)

        assert first == second

    def test_a_clustered_threshold_below_k_clusters_carries_a_few_clusters_warning(self) -> None:
        t = clustered_threshold(clustered_fit(4))

        assert any(w.startswith("FEW_CLUSTERS: ") for w in t.warnings)


class TestNoExactClaimsOnClusteredThresholds:
    """Every exact claim is a cell-level test that assumes independent trials, so a clustered
    threshold makes none: no ``RIGHT``, ``LEFT`` or ``FAILS_AT_BASELINE``."""

    @pytest.mark.parametrize("target", [0.001, 0.01, 0.5, 0.9, 0.999999])
    def test_no_target_produces_an_exact_claim(self, target: float) -> None:
        t = clustered_threshold(target=target)

        assert t.dependence == "clustered"
        assert t.censoring not in (Censoring.RIGHT, Censoring.LEFT)
        assert t.status is not Status.FAILS_AT_BASELINE

    def test_a_beyond_range_target_is_flagged_as_extrapolated_not_exact(self) -> None:
        # With the independence tests this same fit gives censoring RIGHT at this target.
        independent = threshold(
            clustered_fit(dependence="independent"),
            definition=Definition.absolute(0.001),
            interval_method="profile",
            level=0.95,
            dependence="independent",
        )
        assert independent.censoring is Censoring.RIGHT

        t = clustered_threshold(target=0.001)

        assert t.censoring is not Censoring.RIGHT
        assert any(w.startswith("EXTRAPOLATED:") for w in t.warnings) or t.status is not Status.OK


class TestRatioIntervalGuards:
    def _pair(self) -> tuple[Threshold, Threshold]:
        fit_a, fit_b = paired_fits()
        return clustered_threshold(fit_a), clustered_threshold(fit_b)

    @pytest.mark.parametrize("method", ["fieller", "log_delta"])
    def test_paired_on_shared_clusters_returns_a_paired_ratio(self, method: str) -> None:
        t_a, t_b = self._pair()

        ratio = ratio_interval(t_a, t_b, method=method, dependence="paired")

        assert ratio.dependence == "paired"
        assert ratio.method == method
        assert ratio.bootstrap is None

    def test_paired_cluster_bootstrap_records_its_summary(self) -> None:
        t_a, t_b = self._pair()

        ratio = ratio_interval(
            t_a,
            t_b,
            method="cluster_bootstrap",
            dependence="paired",
            n_resamples=19,
            seed=4,
        )

        assert ratio.dependence == "paired"
        assert ratio.method == "cluster_bootstrap"
        assert ratio.bootstrap is not None and ratio.bootstrap.seed == 4

    def test_paired_with_independent_thresholds_raises(self) -> None:
        self._pair()
        t_a = threshold(
            clustered_fit(dependence="independent", base="a"),
            definition=Definition.absolute(0.5),
            interval_method="profile",
            level=0.95,
            dependence="independent",
        )
        t_b = threshold(
            clustered_fit(dependence="independent", base="b"),
            definition=Definition.absolute(0.5),
            interval_method="profile",
            level=0.95,
            dependence="independent",
        )

        with pytest.raises(ValueError):
            ratio_interval(t_a, t_b, method="fieller", dependence="paired")

    def test_paired_with_one_clustered_and_one_independent_threshold_raises(self) -> None:
        t_a, _ = self._pair()
        t_b = threshold(
            clustered_fit(dependence="independent", base="b"),
            definition=Definition.absolute(0.5),
            interval_method="profile",
            level=0.95,
            dependence="independent",
        )

        with pytest.raises(ValueError):
            ratio_interval(t_a, t_b, method="fieller", dependence="paired")

    def test_paired_with_different_cluster_sets_raises(self) -> None:
        t_a, _ = self._pair()
        t_b = clustered_threshold(clustered_fit(prefix="z", base="b"))

        with pytest.raises(ValueError):
            ratio_interval(t_a, t_b, method="fieller", dependence="paired")

    def test_paired_with_overlapping_but_unequal_cluster_sets_raises(self) -> None:
        t_a, _ = self._pair()
        t_b = clustered_threshold(clustered_fit(5, base="b"))

        with pytest.raises(ValueError):
            ratio_interval(t_a, t_b, method="fieller", dependence="paired")

    def test_paired_with_two_thresholds_from_one_fit_raises(self) -> None:
        fit_a, _ = paired_fits()
        t_a = clustered_threshold(fit_a, target=0.5)
        t_b = clustered_threshold(fit_a, target=0.7)

        with pytest.raises(ValueError):
            ratio_interval(t_a, t_b, method="fieller", dependence="paired")

    def test_paired_with_different_definitions_raises(self) -> None:
        fit_a, fit_b = paired_fits()
        t_a = clustered_threshold(fit_a, target=0.5)
        t_b = clustered_threshold(fit_b, target=0.7)

        with pytest.raises(ValueError):
            ratio_interval(t_a, t_b, method="fieller", dependence="paired")

    def test_cluster_bootstrap_with_independent_dependence_raises(self) -> None:
        t_a = clustered_threshold(clustered_fit(prefix="p"))
        t_b = clustered_threshold(clustered_fit(prefix="q", base="b"))
        # The disjoint pair is a valid independent ratio, so the failure below is the method.
        assert ratio_interval(t_a, t_b, method="fieller", dependence="independent").dependence == (
            "independent"
        )

        with pytest.raises(ValueError):
            ratio_interval(
                t_a,
                t_b,
                method="cluster_bootstrap",
                dependence="independent",
                n_resamples=19,
                seed=1,
            )

    @pytest.mark.parametrize("method", ["fieller", "log_delta"])
    @pytest.mark.parametrize("extra", [{"n_resamples": 19}, {"seed": 1}])
    def test_resample_arguments_are_refused_without_the_cluster_bootstrap(
        self, method: str, extra: dict[str, int]
    ) -> None:
        t_a, t_b = self._pair()
        ratio_interval(t_a, t_b, method=method, dependence="paired")

        with pytest.raises(ValueError):
            ratio_interval(t_a, t_b, method=method, dependence="paired", **extra)

    def test_an_unknown_method_raises(self) -> None:
        t_a, t_b = self._pair()

        with pytest.raises(ValueError):
            ratio_interval(t_a, t_b, method="bootstrap", dependence="paired")

    def test_an_unknown_dependence_raises(self) -> None:
        t_a, t_b = self._pair()

        with pytest.raises(ValueError):
            ratio_interval(t_a, t_b, method="fieller", dependence="clustered")

    def test_independent_dependence_on_clustered_thresholds_with_disjoint_clusters_is_allowed(
        self,
    ) -> None:
        t_a = clustered_threshold(clustered_fit(prefix="p"))
        t_b = clustered_threshold(clustered_fit(prefix="q", base="b"))

        ratio = ratio_interval(t_a, t_b, method="log_delta", dependence="independent")

        assert ratio.dependence == "independent"

    def test_independent_dependence_with_overlapping_clusters_still_raises(self) -> None:
        t_a, t_b = self._pair()

        with pytest.raises(ValueError):
            ratio_interval(t_a, t_b, method="fieller", dependence="independent")


class TestDiagnosticsOnClusteredFits:
    def test_stored_diagnostics_equal_diagnose_of_the_fit(self) -> None:
        fit = clustered_fit()

        assert fit.diagnostics is not None
        assert fit.diagnostics == diagnose(fit)

    def test_clustered_fits_warn_that_goodness_of_fit_assumes_independence(self) -> None:
        fit = clustered_fit()

        assert fit.diagnostics is not None
        assert any(w.startswith("CLUSTERED_GOF: ") for w in fit.diagnostics.warnings)

    def test_independent_fits_do_not_carry_the_clustered_gof_token(self) -> None:
        for fit in (clustered_fit(dependence="independent"), fake_fit()):
            assert fit.diagnostics is not None
            assert not any(w.startswith("CLUSTERED_GOF: ") for w in fit.diagnostics.warnings)
