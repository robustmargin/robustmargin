"""Property tests for :func:`~robustmargin.check_monotonicity` (hypothesis).

Two properties, mirroring ``tests/property/test_models_properties.py``'s own conventions:

- Row permutation changes nothing (the check pools per exact severity before testing anything,
  the same way ``Observations.cells()`` already does).
- Aggregated and disaggregated inputs give identical checks.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from robustmargin import Axis, Observations, check_monotonicity

_MAX_EXAMPLES = 25
_AXIS = Axis(name="severity", unit="unit", scale="log")


@st.composite
def _counts_with_optional_control(
    draw: st.DrawFn,
) -> tuple[list[float], list[int], list[int]]:
    has_control = draw(st.booleans())
    n_nonzero = draw(st.integers(min_value=1, max_value=5))
    nonzero_severities = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False),
            min_size=n_nonzero,
            max_size=n_nonzero,
            unique=True,
        )
    )
    severities = ([0.0] if has_control else []) + nonzero_severities
    trials = draw(
        st.lists(
            st.integers(min_value=1, max_value=50),
            min_size=len(severities),
            max_size=len(severities),
        )
    )
    successes = [draw(st.integers(min_value=0, max_value=t)) for t in trials]
    return severities, successes, trials


def _from_counts(counts: tuple[list[float], list[int], list[int]]) -> Observations:
    severity, successes, trials = counts
    return Observations.from_counts(
        _AXIS,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )


def _rows_from_counts(counts: tuple[list[float], list[int], list[int]]) -> Observations:
    severity, successes, trials = counts
    row_severity: list[float] = []
    row_success: list[int] = []
    for level, n_success, n_trials in zip(severity, successes, trials, strict=True):
        row_severity.extend([level] * n_trials)
        row_success.extend([1] * n_success + [0] * (n_trials - n_success))
    return Observations.from_observations(
        _AXIS,
        severity=row_severity,
        success=row_success,
        outcome="success",
        direction="decreasing",
    )


@given(_counts_with_optional_control())
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_aggregated_and_disaggregated_inputs_give_identical_checks(
    counts: tuple[list[float], list[int], list[int]],
) -> None:
    aggregated = _from_counts(counts)
    disaggregated = _rows_from_counts(counts)

    assert check_monotonicity(aggregated) == check_monotonicity(disaggregated)


@given(_counts_with_optional_control(), st.data())
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_row_permutation_does_not_change_the_check(
    counts: tuple[list[float], list[int], list[int]],
    data: st.DataObject,
) -> None:
    severity, successes, trials = counts
    n = len(severity)
    order = data.draw(st.permutations(range(n)))

    original = _from_counts(counts)
    permuted = _from_counts(
        (
            [severity[i] for i in order],
            [successes[i] for i in order],
            [trials[i] for i in order],
        )
    )

    assert check_monotonicity(original) == check_monotonicity(permuted)
