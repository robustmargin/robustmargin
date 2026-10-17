"""Property tests for ``robustmargin.types.Observations`` (hypothesis).

Two invariants: aggregated and disaggregated inputs give the same pooled cells (the same
``.cells()``, so the same fit), and permuting rows changes nothing. A third property exercises the
fakes' round-trip through ``report.to_dict`` / ``from_dict`` across the combinations
``robustmargin.testing`` produces, complementing the exhaustive parametrised check in
``tests/unit/test_testing.py`` with hypothesis's own example generation and shrinking.
"""

from __future__ import annotations

from collections.abc import Callable

from hypothesis import given, settings
from hypothesis import strategies as st

from robustmargin import Axis, Censoring, IntervalShape, Observations, Status
from robustmargin.report import from_dict, to_dict
from robustmargin.testing import fake_fit, fake_ratio, fake_threshold

_MAX_EXAMPLES = 100
_AXIS = Axis(name="noise", unit="m", scale="log")


@st.composite
def _counts(draw: st.DrawFn) -> tuple[list[float], list[int], list[int]]:
    """Draw a valid ``(severity, successes, trials)`` call with distinct severities.

    Severities are drawn distinct so ``.cells()``'s pooling step is a no-op on the aggregated
    side, keeping the ``from_counts`` / ``from_observations`` comparison about row
    representation, not about the pooling rule itself (which ``tests/unit/test_types.py``
    already covers directly).
    """
    n = draw(st.integers(min_value=1, max_value=6))
    severities = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1_000.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    trials = draw(st.lists(st.integers(min_value=1, max_value=50), min_size=n, max_size=n))
    successes = [draw(st.integers(min_value=0, max_value=t)) for t in trials]
    return severities, successes, trials


@given(_counts())
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_aggregated_and_disaggregated_inputs_give_identical_cells(
    counts: tuple[list[float], list[int], list[int]],
) -> None:
    """``from_counts`` and ``from_observations`` of the same data agree on ``.cells()``."""
    severities, successes, trials = counts

    aggregated = Observations.from_counts(
        _AXIS,
        severity=severities,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )

    row_severity: list[float] = []
    row_success: list[int] = []
    for severity, n_success, n_trials in zip(severities, successes, trials, strict=True):
        row_severity.extend([severity] * n_trials)
        row_success.extend([1] * n_success + [0] * (n_trials - n_success))

    disaggregated = Observations.from_observations(
        _AXIS,
        severity=row_severity,
        success=row_success,
        outcome="success",
        direction="decreasing",
    )

    assert aggregated.cells() == disaggregated.cells()


@given(_counts(), st.data())
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_row_permutation_does_not_change_cells(
    counts: tuple[list[float], list[int], list[int]],
    data: st.DataObject,
) -> None:
    severities, successes, trials = counts
    n = len(severities)
    order = data.draw(st.permutations(range(n)))

    original = Observations.from_counts(
        _AXIS,
        severity=severities,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )
    permuted = Observations.from_counts(
        _AXIS,
        severity=[severities[i] for i in order],
        successes=[successes[i] for i in order],
        trials=[trials[i] for i in order],
        outcome="success",
        direction="decreasing",
    )

    assert original.cells() == permuted.cells()


_FAKE_BUILDERS: tuple[Callable[[], object], ...] = (
    lambda: fake_fit(),
    lambda: fake_fit(status=Status.NOT_CONVERGED),
    lambda: fake_threshold(),
    lambda: fake_threshold(status=Status.SEPARATION, censoring=Censoring.RIGHT),
    lambda: fake_threshold(status=Status.UNREACHABLE, censoring=Censoring.NONE),
    lambda: fake_ratio(),
    lambda: fake_ratio(shape=IntervalShape.UNBOUNDED),
)


@given(st.sampled_from(_FAKE_BUILDERS))
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_fakes_round_trip_through_to_dict_and_from_dict(build: Callable[[], object]) -> None:
    fake = build()

    assert from_dict(to_dict(fake)) == fake
