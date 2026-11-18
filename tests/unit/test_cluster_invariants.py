"""Construction-time invariants of the clustered and paired result fields.

Valid bases are real results built through the public API; each test breaks exactly one thing with
``dataclasses.replace`` (which re-runs ``__post_init__``) and expects ``ValueError``. A base that
cannot be built fails the test at setup, so none of these can pass for the wrong reason.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from _cluster_fixtures import clustered_fit, clustered_threshold, paired_fits

from robustmargin import (
    BootstrapSummary,
    Censoring,
    Definition,
    Fit,
    Ratio,
    Status,
    Threshold,
    ratio_interval,
    threshold,
)
from robustmargin.testing import fake_fit

_PAIR_KW: dict[str, Any] = {
    "severity_low": 1.0,
    "severity_high": 2.0,
    "clusters_reversed": 0,
    "clusters_ordered": 4,
    "clusters_tied": 0,
    "clusters_unbalanced": 0,
    "p_value": 0.0625,
    "holm_adjusted_p": 0.0625,
    "flagged": False,
}


def _sign_pair(**overrides: Any) -> Any:
    from robustmargin import ClusterAdjacentPair

    return ClusterAdjacentPair(**{**_PAIR_KW, **overrides})


def _paired_ratio(method: str = "fieller", **kw: Any) -> Ratio:
    fit_a, fit_b = paired_fits()
    t_a = clustered_threshold(fit_a)
    t_b = clustered_threshold(fit_b)
    return ratio_interval(t_a, t_b, method=method, dependence="paired", **kw)


class TestFitDefaults:
    def test_dependence_and_cluster_cells_are_appended_after_cell_estimates(self) -> None:
        names = [f.name for f in dataclasses.fields(Fit)]

        assert names.index("dependence") == names.index("cell_estimates") + 1
        assert names.index("cluster_cells") == names.index("dependence") + 1

    def test_defaults_are_the_unclustered_behaviour(self) -> None:
        fit = fake_fit()

        assert fit.dependence == "independent"
        assert fit.cluster_cells is None

    def test_a_fit_of_clustered_data_declared_independent_stores_no_cluster_cells(self) -> None:
        fit = clustered_fit(dependence="independent")

        assert fit.dependence == "independent"
        assert fit.cluster_ids is not None
        assert fit.cluster_cells is None

    def test_a_clustered_fit_stores_cluster_cells_and_cluster_ids(self) -> None:
        fit = clustered_fit()

        assert fit.dependence == "clustered"
        assert fit.cluster_ids is not None and fit.cluster_cells is not None
        assert {c.cluster for c in fit.cluster_cells} == set(fit.cluster_ids)


class TestFitInvariants:
    def test_unknown_dependence_raises(self) -> None:
        base = clustered_fit(dependence="independent")
        with pytest.raises(ValueError):
            dataclasses.replace(base, dependence="paired")

    def test_clustered_dependence_without_cluster_cells_raises(self) -> None:
        base = clustered_fit()
        with pytest.raises(ValueError):
            dataclasses.replace(base, cluster_cells=None)

    def test_cluster_cells_without_cluster_ids_raises(self) -> None:
        base = clustered_fit()
        with pytest.raises(ValueError):
            dataclasses.replace(base, cluster_ids=None)

    def test_cluster_cells_ids_must_equal_the_set_of_cluster_ids(self) -> None:
        fit = clustered_fit()
        assert fit.cluster_ids is not None
        with pytest.raises(ValueError):
            dataclasses.replace(fit, cluster_ids=(*fit.cluster_ids[:-1], "unrelated"))

    def test_cluster_cell_sums_must_equal_the_pooled_cells(self) -> None:
        fit = clustered_fit()
        assert fit.cluster_cells is not None
        first = fit.cluster_cells[1]
        bumped = dataclasses.replace(first, successes=min(first.trials, first.successes + 1))
        if bumped == first:
            bumped = dataclasses.replace(first, successes=first.successes - 1)
        broken = (fit.cluster_cells[0], bumped, *fit.cluster_cells[2:])

        with pytest.raises(ValueError):
            dataclasses.replace(fit, cluster_cells=broken)

    def test_cluster_cells_out_of_order_raise(self) -> None:
        fit = clustered_fit()
        assert fit.cluster_cells is not None

        with pytest.raises(ValueError):
            dataclasses.replace(fit, cluster_cells=tuple(reversed(fit.cluster_cells)))

    def test_cluster_cells_on_a_non_binomial_fit_raise(self) -> None:
        gaussian = fake_fit(model="gaussian")
        cells = clustered_fit().cluster_cells

        with pytest.raises(ValueError):
            dataclasses.replace(gaussian, cluster_cells=cells)

    def test_clustered_fit_with_an_independent_monotonicity_check_raises(self) -> None:
        independent = clustered_fit(dependence="independent")
        base = clustered_fit()
        assert independent.monotonicity is not None

        with pytest.raises(ValueError):
            dataclasses.replace(base, monotonicity=independent.monotonicity)

    def test_independent_fit_with_a_cluster_sign_test_check_raises(self) -> None:
        clustered = clustered_fit()
        base = clustered_fit(dependence="independent")

        with pytest.raises(ValueError):
            dataclasses.replace(base, monotonicity=clustered.monotonicity)

    def test_clustered_fit_without_a_monotonicity_check_is_accepted(self) -> None:
        fit = dataclasses.replace(clustered_fit(), monotonicity=None)

        assert fit.monotonicity is None


class TestMonotonicityCheckInvariants:
    def _check(self, **overrides: Any) -> Any:
        from robustmargin import MonotonicityCheck

        kwargs: dict[str, Any] = {
            "direction": "decreasing",
            "alpha": 0.05,
            "dependence": "clustered",
            "method": "cluster_sign_test_one_sided_holm",
            "pairs": (_sign_pair(),),
            "any_flagged": False,
        }
        kwargs.update(overrides)
        return MonotonicityCheck(**kwargs)

    def test_a_consistent_clustered_check_is_accepted(self) -> None:
        check = self._check()

        assert check.dependence == "clustered"
        assert check.method == "cluster_sign_test_one_sided_holm"

    def test_sign_test_method_with_independent_dependence_raises(self) -> None:
        with pytest.raises(ValueError):
            self._check(dependence="independent")

    def test_clustered_dependence_with_the_fisher_method_raises(self) -> None:
        with pytest.raises(ValueError):
            self._check(method="fisher_exact_one_sided_holm")

    def test_sign_test_method_with_adjacent_pair_entries_raises(self) -> None:
        from robustmargin import AdjacentPair

        pair = AdjacentPair(
            severity_low=1.0,
            severity_high=2.0,
            successes_low=9,
            trials_low=10,
            successes_high=5,
            trials_high=10,
            p_value=0.9,
            holm_adjusted_p=0.9,
            flagged=False,
        )

        with pytest.raises(ValueError):
            self._check(pairs=(pair,))

    def test_fisher_method_with_cluster_pair_entries_raises(self) -> None:
        with pytest.raises(ValueError):
            self._check(
                dependence="independent",
                method="fisher_exact_one_sided_holm",
                pairs=(_sign_pair(),),
            )

    def test_flagged_must_equal_holm_adjusted_p_below_alpha(self) -> None:
        with pytest.raises(ValueError):
            self._check(pairs=(_sign_pair(flagged=True),), any_flagged=True)

    def test_any_flagged_must_match_the_pairs(self) -> None:
        with pytest.raises(ValueError):
            self._check(any_flagged=True)

    def test_a_flagged_pair_below_alpha_is_accepted(self) -> None:
        pair = _sign_pair(holm_adjusted_p=0.01, p_value=0.01, flagged=True)

        assert self._check(pairs=(pair,), any_flagged=True).any_flagged is True


class TestThresholdInvariants:
    def test_a_clustered_threshold_records_its_dependence_and_fit(self) -> None:
        fit = clustered_fit()
        t = clustered_threshold(fit)

        assert t.dependence == "clustered"
        assert t.fit is fit
        assert t.interval_method == "delta"

    def test_clustered_threshold_on_an_independent_fit_raises(self) -> None:
        t = clustered_threshold()
        independent = clustered_fit(dependence="independent")

        with pytest.raises(ValueError):
            dataclasses.replace(t, fit=independent)

    def test_independent_threshold_on_a_clustered_fit_raises(self) -> None:
        t = clustered_threshold()

        with pytest.raises(ValueError):
            dataclasses.replace(t, dependence="independent")

    @pytest.mark.parametrize("method", ["profile", "parametric_bootstrap", "cell_bootstrap"])
    def test_clustered_threshold_with_an_independence_interval_method_raises(
        self, method: str
    ) -> None:
        t = clustered_threshold()

        with pytest.raises(ValueError):
            dataclasses.replace(t, interval_method=method)

    def test_cluster_bootstrap_requires_a_bootstrap_summary(self) -> None:
        t = clustered_threshold(interval_method="cluster_bootstrap", n_resamples=19, seed=1)
        assert t.bootstrap is not None

        with pytest.raises(ValueError):
            dataclasses.replace(t, bootstrap=None)

    def test_cluster_bootstrap_on_an_independent_threshold_raises(self) -> None:
        t = clustered_threshold(interval_method="cluster_bootstrap", n_resamples=19, seed=1)
        independent_fit = clustered_fit(dependence="independent")
        independent = threshold(
            independent_fit,
            definition=Definition.absolute(0.5),
            interval_method="profile",
            level=0.95,
            dependence="independent",
        )

        with pytest.raises(ValueError):
            dataclasses.replace(
                independent, interval_method="cluster_bootstrap", bootstrap=t.bootstrap
            )

    @pytest.mark.parametrize("censoring", [Censoring.RIGHT, Censoring.LEFT])
    def test_clustered_threshold_with_an_exact_censoring_claim_raises(
        self, censoring: Censoring
    ) -> None:
        t = clustered_threshold()

        with pytest.raises(ValueError):
            dataclasses.replace(t, censoring=censoring, value=None, lo=None, hi=None)

    def test_clustered_threshold_that_fails_at_baseline_raises(self) -> None:
        t = clustered_threshold()

        with pytest.raises(ValueError):
            dataclasses.replace(
                t,
                status=Status.FAILS_AT_BASELINE,
                censoring=Censoring.NONE,
                value=None,
                lo=None,
                hi=None,
            )

    def test_threshold_is_a_frozen_dataclass_with_a_dependence_field(self) -> None:
        assert "dependence" in {f.name for f in dataclasses.fields(Threshold)}


class TestRatioInvariants:
    def test_a_paired_ratio_records_its_dependence_and_method(self) -> None:
        ratio = _paired_ratio()

        assert ratio.dependence == "paired"
        assert ratio.method == "fieller"
        assert ratio.bootstrap is None

    def test_paired_requires_both_thresholds_clustered(self) -> None:
        ratio = _paired_ratio()
        independent = threshold(
            clustered_fit(dependence="independent", base="b"),
            definition=Definition.absolute(0.5),
            interval_method="profile",
            level=0.95,
            dependence="independent",
        )

        with pytest.raises(ValueError):
            dataclasses.replace(ratio, threshold_b=independent)

    def test_paired_requires_equal_cluster_id_sets(self) -> None:
        ratio = _paired_ratio()
        other_clusters = clustered_threshold(clustered_fit(prefix="z", base="b"))

        with pytest.raises(ValueError):
            dataclasses.replace(ratio, threshold_b=other_clusters)

    def test_cluster_bootstrap_requires_paired_dependence(self) -> None:
        ratio = _paired_ratio("cluster_bootstrap", n_resamples=19, seed=2)

        with pytest.raises(ValueError):
            dataclasses.replace(ratio, dependence="independent")

    def test_bootstrap_summary_is_only_allowed_with_the_cluster_bootstrap_method(self) -> None:
        ratio = _paired_ratio("fieller")
        summary = BootstrapSummary(
            n_resamples=19, failed=0, unreachable_low=0, unreachable_high=0, seed=2
        )

        with pytest.raises(ValueError):
            dataclasses.replace(ratio, bootstrap=summary)

    def test_independent_dependence_is_rejected_for_the_cluster_bootstrap_method(self) -> None:
        fit_a = clustered_fit(prefix="p")
        fit_b = clustered_fit(prefix="q", base="b")
        t_a, t_b = clustered_threshold(fit_a), clustered_threshold(fit_b)
        # Disjoint clusters allow "independent" with the analytic methods...
        ratio = ratio_interval(t_a, t_b, method="fieller", dependence="independent")
        assert ratio.dependence == "independent"

        # ...but the bootstrap method is the paired one.
        with pytest.raises(ValueError):
            dataclasses.replace(ratio, method="cluster_bootstrap")

    def test_paired_ratio_on_fits_is_a_real_fit_object_pair(self) -> None:
        ratio = _paired_ratio()

        assert isinstance(ratio.threshold_a.fit, Fit)
        assert ratio.threshold_a.fit is not ratio.threshold_b.fit
