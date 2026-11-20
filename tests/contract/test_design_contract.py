"""Contract test: the design-planning submodule's public surface.

``robustmargin.design`` is a submodule, like ``robustmargin.report`` and ``robustmargin.testing``:
its names are not re-exported from the package root. A consumer sizing a sweep from a pilot fit
imports from it directly. Only public names are used here.

A failing **live** test in this file means a breaking change to what such a consumer depends on.
**Do not edit a live test to make it pass.** Stop and ask.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

import robustmargin
from robustmargin import Axis, Definition, report
from robustmargin.design import WidthPlan, plan_ratio_width, plan_threshold_width, severity_grid

_AXIS = Axis(name="noise", unit="ratio", scale="log")

# Field names and order are the API.
_FIELDS = (
    "kind",
    "axis",
    "severity",
    "trials_per_level",
    "n_clusters",
    "icc",
    "definition",
    "interval_method",
    "ratio_method",
    "dependence",
    "level",
    "n_replicates",
    "seed",
    "n_resamples",
    "width_scale",
    "widths",
    "outcomes",
    "plan_schema_version",
    "library_version",
    "provenance",
)


def _plan(**overrides: Any) -> WidthPlan:
    kwargs: dict[str, Any] = {
        "kind": "threshold",
        "axis": _AXIS,
        "severity": (0.1, 0.2, 0.4, 0.8),
        "trials_per_level": 100,
        "n_clusters": None,
        "icc": 0.0,
        "definition": Definition.absolute(0.5),
        "interval_method": "delta",
        "ratio_method": None,
        "dependence": "independent",
        "level": 0.95,
        "n_replicates": 4,
        "seed": 7,
        "n_resamples": None,
        "width_scale": "log_ratio",
        "widths": (0.5, None, 0.75, None),
        "outcomes": ("OK", "SEPARATION", "OK", "UNBOUNDED"),
        "plan_schema_version": "1",
        "library_version": "0.4.0",
        "provenance": {"pilot": "coarse sweep"},
    }
    kwargs.update(overrides)
    return WidthPlan(**kwargs)


def test_design_submodule_exposes_its_four_symbols() -> None:
    for name in ("severity_grid", "plan_threshold_width", "plan_ratio_width", "WidthPlan"):
        assert hasattr(robustmargin.design, name)
    assert callable(severity_grid)
    assert callable(plan_threshold_width)
    assert callable(plan_ratio_width)


def test_design_names_are_not_re_exported_from_the_package_root() -> None:
    for name in ("severity_grid", "plan_threshold_width", "plan_ratio_width", "WidthPlan"):
        assert name not in robustmargin.__all__
        assert not hasattr(robustmargin, name)


def test_package_root_all_is_unchanged_by_the_design_submodule() -> None:
    assert "design" not in robustmargin.__all__
    assert len(robustmargin.__all__) == len(set(robustmargin.__all__))
    assert robustmargin.__all__ == sorted(robustmargin.__all__)


def test_width_plan_is_a_frozen_dataclass_with_the_contracted_fields_in_order() -> None:
    assert dataclasses.is_dataclass(WidthPlan)
    assert tuple(f.name for f in dataclasses.fields(WidthPlan)) == _FIELDS
    with pytest.raises(dataclasses.FrozenInstanceError):
        _plan().seed = 8  # type: ignore[misc]


def test_width_plan_provenance_schema_and_library_version_have_defaults() -> None:
    kwargs = {f.name: getattr(_plan(), f.name) for f in dataclasses.fields(WidthPlan)}
    for name in ("provenance", "plan_schema_version", "library_version"):
        del kwargs[name]
    built = WidthPlan(**kwargs)
    assert built.provenance == {}
    assert built.plan_schema_version == "1"
    assert built.library_version == robustmargin.__version__


def test_width_plan_is_keyword_only() -> None:
    values = [getattr(_plan(), f.name) for f in dataclasses.fields(WidthPlan)]
    with pytest.raises(TypeError):
        constructor: Any = WidthPlan
        constructor(*values)


def test_width_plan_methods_are_present() -> None:
    result = _plan()
    assert result.quantile(0.5) >= 0.0
    share, se = result.share_at_most(1.0)
    assert 0.0 <= share <= 1.0 and se >= 0.0
    assert result.outcome_counts() == {"OK": 2, "SEPARATION": 1, "UNBOUNDED": 1}


def test_severity_grid_call_shape() -> None:
    grid = severity_grid(_AXIS, lowest=0.01, highest=1.0, n_levels=9)
    assert grid[0] == 0.0
    assert len(grid) == 10


def test_width_plan_round_trips_through_json_with_none_widths_intact() -> None:
    result = _plan()
    text = json.dumps(report.to_dict(result))
    back = report.from_dict(json.loads(text))
    assert isinstance(back, WidthPlan)
    assert back == result
    assert back.widths == (0.5, None, 0.75, None)
    assert back.outcomes == ("OK", "SEPARATION", "OK", "UNBOUNDED")


def test_ratio_plan_round_trips_with_its_methods_and_dependence() -> None:
    result = _plan(
        kind="ratio",
        ratio_method="log_delta",
        dependence="paired",
        n_clusters=10,
        icc=0.25,
        widths=(0.5, 0.6, 0.7, 0.8),
        outcomes=("OK",) * 4,
    )
    back = report.from_dict(json.loads(json.dumps(report.to_dict(result))))
    assert back == result


def test_serialised_plan_keeps_the_field_names_and_version_tags() -> None:
    data = report.to_dict(_plan())
    assert data["type"] == "WidthPlan"
    assert data["plan_schema_version"] == "1"
    for name in _FIELDS:
        assert name in data
    assert data["library_version"] == "0.4.0"
    assert data["provenance"] == {"pilot": "coarse sweep"}


def test_serialised_plan_never_carries_a_non_finite_number() -> None:
    text = json.dumps(report.to_dict(_plan()), allow_nan=False)
    assert "Infinity" not in text and "NaN" not in text


def test_an_unrecognised_plan_schema_version_is_rejected() -> None:
    data = report.to_dict(_plan())
    data["plan_schema_version"] = "99"
    with pytest.raises(ValueError):
        report.from_dict(data)


def test_a_plan_with_an_unknown_field_is_rejected_on_read() -> None:
    data = report.to_dict(_plan())
    data["extra_field"] = 1
    with pytest.raises(ValueError):
        report.from_dict(data)


def test_scorecard_schema_version_is_unchanged_by_the_plan_type() -> None:
    assert report.SCHEMA_VERSION == "3"


def test_a_scorecard_carrying_a_width_plan_is_rejected_on_read() -> None:
    from robustmargin import Scorecard
    from robustmargin.testing import fake_fit

    data = report.to_dict(Scorecard(results=(fake_fit(),), provenance={}))
    data["results"] = [report.to_dict(_plan())]
    with pytest.raises(ValueError):
        report.from_dict(data)
