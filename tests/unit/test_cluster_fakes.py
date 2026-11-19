"""``robustmargin.testing`` fakes for clustered and paired results.

A fake that is not schema-valid, or not internally consistent with the real constructors, would
let a consumer's own tests pass against a shape the library could never produce.
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema
import pytest

from robustmargin import Censoring, IntervalShape, Status, diagnose
from robustmargin.report import from_dict, load_schema, to_dict
from robustmargin.testing import fake_fit, fake_ratio, fake_threshold


def _validator(type_name: str) -> jsonschema.protocols.Validator:
    schema = load_schema()
    return jsonschema.Draft202012Validator(
        {"$defs": schema["$defs"], "$ref": f"#/$defs/{type_name}"}
    )


def _json(obj: object) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(json.dumps(to_dict(obj), allow_nan=False))
    return payload


class TestFakeFit:
    def test_clustered_fake_is_a_clustered_binomial_fit_with_cluster_cells(self) -> None:
        fit = fake_fit(dependence="clustered")

        assert fit.dependence == "clustered"
        assert fit.model == "binomial"
        assert fit.cluster_ids is not None
        assert fit.cluster_cells is not None
        assert {c.cluster for c in fit.cluster_cells} == set(fit.cluster_ids)

    def test_clustered_fake_carries_a_cluster_sign_test_check(self) -> None:
        fit = fake_fit(dependence="clustered")

        assert fit.monotonicity is not None
        assert fit.monotonicity.dependence == "clustered"
        assert fit.monotonicity.method == "cluster_sign_test_one_sided_holm"

    def test_clustered_fake_is_schema_valid_and_round_trips(self) -> None:
        fit = fake_fit(dependence="clustered")

        _validator("Fit").validate(_json(fit))
        assert from_dict(_json(fit)) == fit

    def test_clustered_fake_diagnostics_equal_diagnose_with_the_clustered_gof_token(self) -> None:
        fit = fake_fit(dependence="clustered")

        assert fit.diagnostics is not None
        assert fit.diagnostics == diagnose(fit)
        assert any(w.startswith("CLUSTERED_GOF: ") for w in fit.diagnostics.warnings)

    def test_default_dependence_is_independent_with_no_cluster_cells(self) -> None:
        fit = fake_fit()

        assert fit.dependence == "independent"
        assert fit.cluster_cells is None

    @pytest.mark.parametrize("model", ["gaussian", "isotonic"])
    def test_clustered_fake_raises_for_a_non_binomial_model(self, model: str) -> None:
        assert fake_fit(dependence="clustered").dependence == "clustered"

        with pytest.raises(ValueError):
            fake_fit(model=model, dependence="clustered")

    def test_an_unknown_dependence_raises(self) -> None:
        with pytest.raises(ValueError):
            fake_fit(dependence="paired")


class TestFakeThreshold:
    def test_clustered_fake_defaults_to_the_cluster_bootstrap_with_a_summary(self) -> None:
        t = fake_threshold(dependence="clustered")

        assert t.dependence == "clustered"
        assert t.interval_method == "cluster_bootstrap"
        assert t.bootstrap is not None
        assert t.fit.dependence == "clustered"

    def test_clustered_fake_is_schema_valid_and_round_trips(self) -> None:
        t = fake_threshold(dependence="clustered")

        _validator("Threshold").validate(_json(t))
        assert from_dict(_json(t)) == t

    def test_clustered_fake_accepts_the_delta_method(self) -> None:
        t = fake_threshold(dependence="clustered", interval_method="delta")

        assert t.interval_method == "delta"
        assert t.bootstrap is None

    @pytest.mark.parametrize("method", ["profile", "parametric_bootstrap", "cell_bootstrap"])
    def test_clustered_fake_refuses_independence_interval_methods(self, method: str) -> None:
        assert fake_threshold(dependence="clustered").dependence == "clustered"

        with pytest.raises(ValueError):
            fake_threshold(dependence="clustered", interval_method=method)

    @pytest.mark.parametrize(
        ("status", "censoring"),
        [
            (Status.FAILS_AT_BASELINE, Censoring.NONE),
            (Status.OK, Censoring.RIGHT),
            (Status.OK, Censoring.LEFT),
        ],
    )
    def test_clustered_fake_refuses_exact_claims(
        self, status: Status, censoring: Censoring
    ) -> None:
        assert fake_threshold(dependence="clustered").dependence == "clustered"

        with pytest.raises(ValueError):
            fake_threshold(dependence="clustered", status=status, censoring=censoring)

    def test_default_dependence_is_independent(self) -> None:
        assert fake_threshold().dependence == "independent"

    def test_an_unknown_dependence_raises(self) -> None:
        with pytest.raises(ValueError):
            fake_threshold(dependence="paired")


class TestFakeRatio:
    def test_paired_fake_defaults_to_a_bounded_cluster_bootstrap_ratio(self) -> None:
        ratio = fake_ratio(dependence="paired")

        assert ratio.dependence == "paired"
        assert ratio.method == "cluster_bootstrap"
        assert ratio.shape is IntervalShape.BOUNDED
        assert ratio.bootstrap is not None

    def test_paired_fake_is_two_clustered_fits_on_a_shared_cluster_set(self) -> None:
        ratio = fake_ratio(dependence="paired")
        fit_a, fit_b = ratio.threshold_a.fit, ratio.threshold_b.fit

        assert fit_a.dependence == "clustered" and fit_b.dependence == "clustered"
        assert fit_a is not fit_b
        assert fit_a.cluster_ids is not None and fit_b.cluster_ids is not None
        assert set(fit_a.cluster_ids) == set(fit_b.cluster_ids)

    def test_paired_fake_is_schema_valid_and_round_trips(self) -> None:
        ratio = fake_ratio(dependence="paired")

        _validator("Ratio").validate(_json(ratio))
        assert from_dict(_json(ratio)) == ratio

    def test_default_dependence_stays_independent_and_has_no_bootstrap(self) -> None:
        ratio = fake_ratio()

        assert ratio.dependence == "independent"
        assert ratio.bootstrap is None

    def test_an_unknown_dependence_raises(self) -> None:
        with pytest.raises(ValueError):
            fake_ratio(dependence="clustered")
