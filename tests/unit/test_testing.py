"""Unit tests for ``robustmargin.testing``: ``fake_fit``, ``fake_threshold`` and ``fake_ratio``.

Every combination plan section 5.5 allows must produce a schema-valid, round-tripping object;
every combination it forbids must raise ``ValueError``. A fake that is not schema-valid would
let a consumer's own tests (marginbench's, per plan section 3.3 "unblock marginbench early")
pass against a shape the real implementation could never produce.

``fake_ratio``'s parametrised combinations are a representative sample, not an exhaustive
status x censoring x shape grid like ``fake_threshold``'s: unlike ``Threshold``, ``Ratio``'s
invariants (``tests/unit/test_results.py``) place no joint restriction between ``status`` and
``censoring`` beyond "estimate is ``None`` unless both are trivial", so there is no natural
forbidden combination to enumerate here without inventing one -- and Phase 6 owns ratio
censoring propagation, not this phase.
"""

from __future__ import annotations

import dataclasses
import math

import jsonschema
import pytest

from robustmargin import (
    Censoring,
    Fit,
    IntervalShape,
    Observations,
    Status,
    check_monotonicity,
    diagnose,
)
from robustmargin.report import from_dict, load_schema, to_dict
from robustmargin.testing import fake_fit, fake_ratio, fake_threshold

# Round 2, rule 9: Fit.status may never be UNREACHABLE or FAILS_AT_BASELINE (those describe
# whether a *threshold* can be solved given a successful fit, not whether the fit itself
# succeeded), so fake_fit's own status parametrisation must exclude them.
_FIT_STATUSES: tuple[Status, ...] = tuple(
    status for status in Status if status not in (Status.UNREACHABLE, Status.FAILS_AT_BASELINE)
)

_ALLOWED_THRESHOLD_COMBINATIONS: tuple[tuple[Status, Censoring], ...] = (
    *((Status.OK, censoring) for censoring in Censoring),
    *(
        (status, censoring)
        for status in (Status.SEPARATION, Status.NOT_CONVERGED)
        for censoring in (Censoring.NONE, Censoring.RIGHT, Censoring.LEFT)
    ),
    *(
        (status, Censoring.NONE)
        for status in (Status.UNREACHABLE, Status.FAILS_AT_BASELINE, Status.CONTROL_INCOMPATIBLE)
    ),
)

_FORBIDDEN_THRESHOLD_COMBINATIONS: tuple[tuple[Status, Censoring], ...] = (
    (Status.UNREACHABLE, Censoring.RIGHT),
    (Status.FAILS_AT_BASELINE, Censoring.LEFT),
    (Status.CONTROL_INCOMPATIBLE, Censoring.OPEN_UPPER),
    (Status.SEPARATION, Censoring.OPEN_UPPER),
    (Status.NOT_CONVERGED, Censoring.OPEN_LOWER),
)

_VALID_RATIO_COMBINATIONS: tuple[tuple[IntervalShape | None, Status, Censoring], ...] = (
    (IntervalShape.BOUNDED, Status.OK, Censoring.NONE),
    (IntervalShape.UNBOUNDED, Status.OK, Censoring.NONE),
    (IntervalShape.EXCLUSIVE, Status.OK, Censoring.NONE),
    (None, Status.OK, Censoring.NONE),
    # Round 3, item 2: an *explicit* shape is only valid for status OK + censoring NONE, so the
    # non-OK / non-NONE-censoring combinations here use shape=None (fake_ratio's own default),
    # not an explicit IntervalShape member as in earlier rounds.
    (None, Status.NOT_CONVERGED, Censoring.NONE),
    (None, Status.OK, Censoring.RIGHT),
)


def _validator_for(type_name: str) -> jsonschema.protocols.Validator:
    schema = load_schema()
    return jsonschema.Draft202012Validator(
        {"$defs": schema["$defs"], "$ref": f"#/$defs/{type_name}"}
    )


class TestFakeFit:
    def test_default_status_is_ok(self) -> None:
        fit = fake_fit()

        assert fit.status is Status.OK

    @pytest.mark.parametrize("status", _FIT_STATUSES)
    def test_every_status_produces_a_schema_valid_object(self, status: Status) -> None:
        fit = fake_fit(status=status)

        _validator_for("Fit").validate(to_dict(fit))

    @pytest.mark.parametrize("status", _FIT_STATUSES)
    def test_every_status_round_trips(self, status: Status) -> None:
        fit = fake_fit(status=status)

        assert from_dict(to_dict(fit)) == fit

    @pytest.mark.parametrize("status", [Status.UNREACHABLE, Status.FAILS_AT_BASELINE])
    def test_threshold_level_status_is_rejected(self, status: Status) -> None:
        """Direct consequence of rule 9 (``Fit.status`` may not be ``UNREACHABLE`` or
        ``FAILS_AT_BASELINE``): ``fake_fit`` must not silently build an invalid ``Fit`` for
        either."""
        with pytest.raises(ValueError):
            fake_fit(status=status)

    def test_warnings_mentions_fake(self) -> None:
        fit = fake_fit()

        assert any("fake" in warning.lower() for warning in fit.warnings)

    def test_provenance_override_is_stored(self) -> None:
        fit = fake_fit(provenance={"note": "test"})

        assert fit.provenance == {"note": "test"}


def _severities_strictly_increase(fit: Fit) -> bool:
    severities = [cell.severity for cell in fit.cells]
    return all(a < b for a, b in zip(severities, severities[1:], strict=False))


class TestFakeCellsAreStrictlyAscending:
    """decisions/0029: ``Fit`` rejects ``cells`` that are not strictly ascending in severity, and
    the fakes are what a consumer builds its own fixtures from. Each must keep producing sorted,
    duplicate-free cells -- on every status, and in the fits nested inside a threshold or a
    ratio. (Were a fake to drift, ``Fit`` would already refuse to construct it; these tests say
    what the property is, and that more than one cell is exercised, so a one-cell fake could
    never satisfy them vacuously.)"""

    @pytest.mark.parametrize("status", _FIT_STATUSES)
    def test_fake_fit_cells_are_strictly_ascending_on_every_status(self, status: Status) -> None:
        fit = fake_fit(status=status)

        assert len(fit.cells) >= 2
        assert _severities_strictly_increase(fit)

    @pytest.mark.parametrize(("status", "censoring"), _ALLOWED_THRESHOLD_COMBINATIONS)
    def test_fake_threshold_fit_cells_are_strictly_ascending(
        self, status: Status, censoring: Censoring
    ) -> None:
        threshold = fake_threshold(status=status, censoring=censoring)

        assert len(threshold.fit.cells) >= 2
        assert _severities_strictly_increase(threshold.fit)

    def test_fake_ratio_operand_fit_cells_are_strictly_ascending(self) -> None:
        ratio = fake_ratio()

        for operand in (ratio.threshold_a, ratio.threshold_b):
            assert len(operand.fit.cells) >= 2
            assert _severities_strictly_increase(operand.fit)

    @pytest.mark.parametrize("status", _FIT_STATUSES)
    def test_fake_fit_cells_survive_a_json_round_trip_in_order(self, status: Status) -> None:
        fit = fake_fit(status=status)

        rebuilt = from_dict(to_dict(fit))

        assert isinstance(rebuilt, Fit)
        assert rebuilt.cells == fit.cells


def _observations_from_fit_cells(fit: Fit) -> Observations:
    return Observations.from_counts(
        fit.axis,
        severity=[c.severity for c in fit.cells],
        successes=[c.successes for c in fit.cells],
        trials=[c.trials for c in fit.cells],
        outcome=fit.outcome,
        direction=fit.direction,
    )


class TestFakeFitMirrorsARealFit:
    """decisions/0023 Amendment 2, item 4: ``fake_fit`` fills ``monotonicity`` through
    :func:`~robustmargin.check_monotonicity` on its own cells for *every* status, and
    ``diagnostics`` through :func:`~robustmargin.diagnose` when ``status`` is ``OK`` -- the same way
    ``fake_threshold`` already builds ``baseline`` through the real
    ``per_cell_clopper_pearson`` path rather than a hand-typed placeholder. Explicit
    ``diagnostics=``/``monotonicity=`` arguments still override (unchanged from before this
    amendment)."""

    @pytest.mark.parametrize("status", _FIT_STATUSES)
    def test_monotonicity_is_not_none_for_every_status(self, status: Status) -> None:
        fit = fake_fit(status=status)

        assert fit.monotonicity is not None

    @pytest.mark.parametrize("status", _FIT_STATUSES)
    def test_monotonicity_matches_check_monotonicity_on_the_same_cells(
        self, status: Status
    ) -> None:
        fit = fake_fit(status=status)

        assert fit.monotonicity == check_monotonicity(_observations_from_fit_cells(fit))

    def test_diagnostics_is_not_none_when_status_is_ok(self) -> None:
        fit = fake_fit(status=Status.OK)

        assert fit.diagnostics is not None

    @pytest.mark.parametrize("status", [s for s in _FIT_STATUSES if s is not Status.OK])
    def test_diagnostics_is_none_when_status_is_not_ok(self, status: Status) -> None:
        fit = fake_fit(status=status)

        assert fit.diagnostics is None

    def test_diagnostics_matches_diagnose_on_the_same_fit(self) -> None:
        fit = fake_fit(status=Status.OK)

        assert fit.diagnostics == diagnose(fit)

    def test_explicit_diagnostics_argument_still_overrides(self) -> None:
        # Overriding `warnings` (rather than a GOF field) sidesteps Diagnostics's own
        # cross-field invariants entirely -- this test only needs a distinguishable sentinel,
        # not a semantically meaningful one.
        real = diagnose(fake_fit(status=Status.OK))
        sentinel = dataclasses.replace(real, warnings=(*real.warnings, "override marker"))

        fit = fake_fit(status=Status.OK, diagnostics=sentinel)

        assert fit.diagnostics == sentinel

    def test_explicit_monotonicity_argument_still_overrides(self) -> None:
        real = check_monotonicity(_observations_from_fit_cells(fake_fit()))
        sentinel = dataclasses.replace(real, warnings=(*real.warnings, "override marker"))

        fit = fake_fit(monotonicity=sentinel)

        assert fit.monotonicity == sentinel


class TestFakeThreshold:
    @pytest.mark.parametrize(("status", "censoring"), _ALLOWED_THRESHOLD_COMBINATIONS)
    def test_allowed_combination_is_schema_valid_and_round_trips(
        self, status: Status, censoring: Censoring
    ) -> None:
        threshold = fake_threshold(status=status, censoring=censoring)

        _validator_for("Threshold").validate(to_dict(threshold))
        assert from_dict(to_dict(threshold)) == threshold

    @pytest.mark.parametrize(("status", "censoring"), _ALLOWED_THRESHOLD_COMBINATIONS)
    def test_allowed_combination_warnings_mentions_fake(
        self, status: Status, censoring: Censoring
    ) -> None:
        threshold = fake_threshold(status=status, censoring=censoring)

        assert any("fake" in warning.lower() for warning in threshold.warnings)

    @pytest.mark.parametrize(("status", "censoring"), _FORBIDDEN_THRESHOLD_COMBINATIONS)
    def test_forbidden_combination_raises_value_error(
        self, status: Status, censoring: Censoring
    ) -> None:
        with pytest.raises(ValueError):
            fake_threshold(status=status, censoring=censoring)

    def test_default_is_status_ok_censoring_none(self) -> None:
        threshold = fake_threshold()

        assert threshold.status is Status.OK
        assert threshold.censoring is Censoring.NONE

    def test_provenance_override_is_stored(self) -> None:
        threshold = fake_threshold(provenance={"note": "test"})

        assert threshold.provenance == {"note": "test"}


class TestFakeRatio:
    @pytest.mark.parametrize(("shape", "status", "censoring"), _VALID_RATIO_COMBINATIONS)
    def test_valid_combination_is_schema_valid_and_round_trips(
        self, shape: IntervalShape | None, status: Status, censoring: Censoring
    ) -> None:
        ratio = fake_ratio(shape=shape, status=status, censoring=censoring)

        _validator_for("Ratio").validate(to_dict(ratio))
        assert from_dict(to_dict(ratio)) == ratio

    @pytest.mark.parametrize(("shape", "status", "censoring"), _VALID_RATIO_COMBINATIONS)
    def test_valid_combination_warnings_mentions_fake(
        self, shape: IntervalShape | None, status: Status, censoring: Censoring
    ) -> None:
        ratio = fake_ratio(shape=shape, status=status, censoring=censoring)

        assert any("fake" in warning.lower() for warning in ratio.warnings)

    def test_default_is_bounded_status_ok_censoring_none(self) -> None:
        ratio = fake_ratio()

        assert ratio.shape is IntervalShape.BOUNDED
        assert ratio.status is Status.OK
        assert ratio.censoring is Censoring.NONE

    def test_provenance_override_is_stored(self) -> None:
        ratio = fake_ratio(provenance={"note": "test"})

        assert ratio.provenance == {"note": "test"}

    def test_shape_none_defaults_to_bounded_when_ok_and_none(self) -> None:
        """Item 2, round 3: ``shape=None`` (the new default) means ``BOUNDED`` specifically
        when the ratio is status OK + censoring NONE -- the one combination that requires a
        shape at all."""
        ratio = fake_ratio(shape=None, status=Status.OK, censoring=Censoring.NONE)

        assert ratio.shape is IntervalShape.BOUNDED

    def test_shape_none_stays_none_when_not_ok_and_none(self) -> None:
        ratio = fake_ratio(shape=None, status=Status.NOT_CONVERGED, censoring=Censoring.NONE)

        assert ratio.shape is None

    def test_explicit_shape_with_non_ok_status_raises(self) -> None:
        with pytest.raises(ValueError):
            fake_ratio(
                shape=IntervalShape.BOUNDED, status=Status.NOT_CONVERGED, censoring=Censoring.NONE
            )

    def test_explicit_shape_with_censoring_raises(self) -> None:
        with pytest.raises(ValueError):
            fake_ratio(shape=IntervalShape.BOUNDED, status=Status.OK, censoring=Censoring.RIGHT)


class TestFakeRatioThresholdsComeFromDisjointData:
    """Rule 16: ``fake_ratio``'s two thresholds must come from fits whose ``cells`` differ, so
    they are built from disjoint data -- plan section 5.6 forbids ``dependence="independent"``
    on shared data, and a fake built to demonstrate the independent path should not itself be
    an example of the case that path forbids."""

    def test_threshold_a_and_b_fits_have_different_cells(self) -> None:
        ratio = fake_ratio()

        assert ratio.threshold_a.fit.cells != ratio.threshold_b.fit.cells


class TestFakeRatioSatisfiesOkNoneInvariants:
    """Rule 17: every shape ``fake_ratio`` produces with ``status`` OK / ``censoring`` NONE
    must satisfy rules 11-14 (``tests/unit/test_results.py``)."""

    @pytest.mark.parametrize("shape", list(IntervalShape))
    def test_shape_satisfies_the_ok_none_invariants(self, shape: IntervalShape) -> None:
        ratio = fake_ratio(shape=shape, status=Status.OK, censoring=Censoring.NONE)

        assert ratio.threshold_a.status is Status.OK
        assert ratio.threshold_a.value is not None
        assert ratio.threshold_b.status is Status.OK
        assert ratio.threshold_b.value is not None
        assert ratio.estimate is not None
        assert math.isclose(
            ratio.estimate,
            ratio.threshold_a.value / ratio.threshold_b.value,
            rel_tol=1e-9,
        )

    def test_exclusive_fake_carries_an_estimate_outside_lo_and_hi(self) -> None:
        ratio = fake_ratio(
            shape=IntervalShape.EXCLUSIVE, status=Status.OK, censoring=Censoring.NONE
        )

        assert ratio.lo is not None
        assert ratio.hi is not None
        assert ratio.estimate is not None
        assert ratio.estimate <= ratio.lo or ratio.estimate >= ratio.hi


class TestFakeFitClusterIds:
    """Item 4, round 3: ``fake_fit`` has no cluster information to invent, so it always sets
    ``cluster_ids=None``."""

    def test_cluster_ids_is_none(self) -> None:
        fit = fake_fit()

        assert fit.cluster_ids is None


class TestFakeThresholdDependence:
    """Item 4, round 3: every fake threshold is built from a single fit with no cluster ids, so
    it is honestly labelled ``dependence="independent"``."""

    def test_dependence_is_independent(self) -> None:
        threshold = fake_threshold()

        assert threshold.dependence == "independent"
