"""Reference-value parity for the ``system_c_axis_1`` monotonicity reversal in
``tests/data/pooled_counts.csv``.

Two independent cross-checks:

1. **``scipy.stats.fisher_exact`` directly** (``TestAgainstScipyDirectly``) -- needs no fixture,
   and pins the exact table/alternative convention robustmargin's own ``check_monotonicity`` must
   reproduce.
2. **R's own ``fisher.test(..., alternative="greater")``**
   (``TestAgainstRsFisherTest``), read from the fixture ``generate.R`` emits
   (``tools/drc_reference/monotonicity_system_c_axis_1.json``). A missing fixture fails with a
   clear ``FileNotFoundError`` -- that failure means "regenerate the fixture" with the pinned R
   container (``tools/drc_reference/README.md``), not a statistical finding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from scipy.stats import fisher_exact

from robustmargin import Axis, Observations, check_monotonicity

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tools" / "drc_reference"
_REL_TOL = 1e-4

# tests/data/pooled_counts.csv, series system_c_axis_1, the two lowest severities.
_SEVERITY_LOW = 0.0
_SEVERITY_HIGH = 0.01
_SUCCESSES_LOW, _TRIALS_LOW = 57, 200
_SUCCESSES_HIGH, _TRIALS_HIGH = 143, 300


def _load(name: str) -> dict[str, Any]:
    with (_FIXTURE_DIR / name).open() as fh:
        result: dict[str, Any] = json.load(fh)
    return result


def _check() -> Any:
    axis = Axis(name="axis_1", unit="m", scale="log")
    obs = Observations.from_counts(
        axis,
        severity=[_SEVERITY_LOW, _SEVERITY_HIGH, 0.05, 0.1],
        successes=[_SUCCESSES_LOW, _SUCCESSES_HIGH, 0, 0],
        trials=[_TRIALS_LOW, _TRIALS_HIGH, 300, 300],
        outcome="success",
        direction="decreasing",
    )
    return check_monotonicity(obs)


def _first_pair_p_value() -> float:
    check = _check()
    first = check.pairs[0]
    assert (first.severity_low, first.severity_high) == (_SEVERITY_LOW, _SEVERITY_HIGH)
    return float(first.p_value)


class TestAgainstScipyDirectly:
    def test_raw_p_value_matches_scipy_fisher_exact(self) -> None:
        failures_low = _TRIALS_LOW - _SUCCESSES_LOW
        failures_high = _TRIALS_HIGH - _SUCCESSES_HIGH
        # robustmargin's own table/alternative convention.
        table = [[_SUCCESSES_HIGH, failures_high], [_SUCCESSES_LOW, failures_low]]
        _, expected_p = fisher_exact(table, alternative="greater")

        actual_p = _first_pair_p_value()

        assert actual_p == pytest.approx(float(expected_p), rel=1e-9)

    def test_raw_p_value_is_roughly_1_2e_5(self) -> None:
        """The worked example: the raw one-sided p-value is about 1.2e-5."""
        actual_p = _first_pair_p_value()

        assert actual_p == pytest.approx(1.2e-5, rel=0.1)


class TestAgainstRsFisherTest:
    """Reads ``tools/drc_reference/monotonicity_system_c_axis_1.json``, written by
    ``generate.R``. If the fixture is missing this class fails with ``FileNotFoundError``, not
    a wrong number."""

    def test_r_native_p_value_matches_check_monotonicity(self) -> None:
        fixture = _load("monotonicity_system_c_axis_1.json")
        expected_p = fixture["fisher_test"]["p_value"]

        actual_p = _first_pair_p_value()

        assert actual_p == pytest.approx(expected_p, rel=_REL_TOL)

    def test_r_fixture_dataset_matches_the_csv_counts(self) -> None:
        fixture = _load("monotonicity_system_c_axis_1.json")
        dataset = fixture["dataset"]

        assert dataset["successes_low"] == _SUCCESSES_LOW
        assert dataset["trials_low"] == _TRIALS_LOW
        assert dataset["successes_high"] == _SUCCESSES_HIGH
        assert dataset["trials_high"] == _TRIALS_HIGH
