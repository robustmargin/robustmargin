"""Isotonic ``cell_estimates`` against weighted pool-adjacent-violators solutions.

**Reads only the committed ``tools/drc_reference/isotonic_pava.json``. Never runs R.**

Provenance, stated plainly: the fixture's expected values are **hand-computed** (``"source":
"hand-computed"``), with the pooling written out per case in the file. They were not produced by R.
``generate.R`` holds a block that would regenerate them with ``Iso::pava``; it has not been run.
When it is, the fixture's ``source`` changes and these tests keep reading it unchanged. An expected
value is never adjusted to match an implementation: a mismatch is a finding.

Each case is checked on both input types. The counts are ``successes`` of ``trials`` with weights
``trials``; the continuous cells carry the same rates as means with ``n = weights``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from robustmargin import (
    Axis,
    ContinuousObservations,
    Observations,
    Status,
    fit_dose_response,
)

_FIXTURE = Path(__file__).resolve().parents[2] / "tools" / "drc_reference" / "isotonic_pava.json"
_AXIS = Axis(name="severity", unit="unit", scale="log")
_REL_TOL = 1e-9


def _load() -> dict[str, Any]:
    with _FIXTURE.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


_FIXTURE_DATA = _load()
_CASES: list[dict[str, Any]] = _FIXTURE_DATA["cases"]
_IDS = [case["label"] for case in _CASES]


def test_the_fixture_states_where_its_values_came_from() -> None:
    assert _FIXTURE_DATA["source"] in ("hand-computed", "Iso::pava")
    assert len(_CASES) >= 3
    assert {case["direction"] for case in _CASES} == {"decreasing", "increasing"}


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_counts_cell_estimates_match_the_weighted_pava_solution(case: dict[str, Any]) -> None:
    obs = Observations.from_counts(
        _AXIS,
        severity=_FIXTURE_DATA["severity"],
        successes=case["successes"],
        trials=case["trials"],
        outcome="outcome",
        direction=case["direction"],
    )

    fit = fit_dose_response(obs, model="isotonic")

    assert fit.status is Status.OK
    assert fit.cell_estimates == pytest.approx(case["expected"], rel=_REL_TOL)


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_continuous_cell_estimates_match_the_weighted_pava_solution(case: dict[str, Any]) -> None:
    obs = ContinuousObservations.from_summaries(
        _AXIS,
        severity=_FIXTURE_DATA["severity"],
        mean=case["y"],
        sd=[1.0] * len(case["y"]),
        n=case["weights"],
        outcome="response",
        direction=case["direction"],
    )

    fit = fit_dose_response(obs, model="isotonic")

    assert fit.status is Status.OK
    assert fit.cell_estimates == pytest.approx(case["expected"], rel=_REL_TOL)


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_the_fixture_solution_is_monotone_and_preserves_the_weighted_mean(
    case: dict[str, Any],
) -> None:
    expected = case["expected"]
    weights = case["weights"]
    sign = -1.0 if case["direction"] == "decreasing" else 1.0

    assert all(sign * (b - a) >= -1e-15 for a, b in zip(expected, expected[1:], strict=False))
    total = sum(weights)
    assert sum(w * e for w, e in zip(weights, expected, strict=True)) / total == pytest.approx(
        sum(w * y for w, y in zip(weights, case["y"], strict=True)) / total, rel=1e-12
    )
