"""Property tests for :func:`~robustmargin.diagnose` (hypothesis).

Three properties, mirroring ``tests/property/test_models_properties.py``'s own conventions and
strategies rather than introducing new ones:

- Row permutation changes nothing (``diagnose`` is a pure function of ``fit.cells``, which
  ``Observations.cells()`` already pools and sorts regardless of row order).
- Aggregated and disaggregated inputs give identical diagnostics, for the same reason.
- With ``u`` fixed at 1, adding an all-success control row leaves ``deviance``,
  ``pearson_chi2``, ``df`` and ``dispersion`` unchanged (the control-row property, extended to
  diagnostics). This is the property that pins down a design decision: the control (a cell whose
  fitted success probability is exactly ``1.0``, under ``upper=1.0`` fixed) is excluded from
  those four statistics, the same way the R fixture's ``glm`` is fit without that row (see
  ``tests/unit/test_validation.py``'s module docstring and
  ``tests/reference/test_diagnostics_drc.py``). ``link_aic``/``link_status`` are **not**
  asserted equal here: adding a row can change which alternative links converge, so this
  property is scoped to the four GOF/dispersion fields only. (``Fit.monotonicity`` is a
  separate field entirely -- never touched by ``Diagnostics`` or by this file; its own
  aggregation/permutation properties live in ``tests/property/test_monotonicity_properties.py``.)

Examples are kept small so real optimizer calls stay fast.
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from robustmargin import Axis, Fit, Observations, Status, diagnose, fit_dose_response

_MAX_EXAMPLES = 25
_AXIS = Axis(name="severity", unit="unit", scale="log")


@st.composite
def _counts_with_optional_control(
    draw: st.DrawFn,
) -> tuple[list[float], list[int], list[int]]:
    """Every nonzero level carries at least one success and one failure -- the same
    separation-avoiding shape as ``_non_separated_counts`` below (and
    ``tests/property/test_models_properties.py``'s identical strategy) -- with an optional
    all-success control row at severity 0.0 prepended. The control is excluded from the
    separation check entirely regardless of its own counts (a log-axis zero-severity control
    carries no covariate information), so prepending it here can never introduce separation on
    its own.

    Drawing ``successes`` freely in ``[0, trials]`` at every level, including the nonzero ones,
    would routinely produce separated (and occasionally non-converged) fits and trip
    Hypothesis's ``filter_too_much`` health check on the ``assume(status is OK)`` calls below.
    Narrowing the *generator* here (rather than suppressing the health check) avoids that: a fit
    built from this strategy converges to ``Status.OK`` in the
    overwhelming majority of draws, so ``assume`` only ever discards the rare, genuine
    non-convergence (for example a wrong-sign pilot slope), not a systematic fraction of them.
    """
    has_control = draw(st.booleans())
    n_nonzero = draw(st.integers(min_value=2, max_value=4))
    nonzero_severities = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False),
            min_size=n_nonzero,
            max_size=n_nonzero,
            unique=True,
        )
    )
    nonzero_trials = draw(
        st.lists(
            st.integers(min_value=5, max_value=50),
            min_size=n_nonzero,
            max_size=n_nonzero,
        )
    )
    nonzero_successes = [draw(st.integers(min_value=1, max_value=t - 1)) for t in nonzero_trials]

    if not has_control:
        return nonzero_severities, nonzero_successes, nonzero_trials

    control_trials = draw(st.integers(min_value=1, max_value=50))
    severities = [0.0, *nonzero_severities]
    successes = [control_trials, *nonzero_successes]
    trials = [control_trials, *nonzero_trials]
    return severities, successes, trials


@st.composite
def _non_separated_counts(draw: st.DrawFn) -> tuple[list[float], list[int], list[int]]:
    """Every level carries at least one success and one failure, which provably rules out
    the pre-fit separation check (see the identical strategy and reasoning in
    ``tests/property/test_models_properties.py``)."""
    n = draw(st.integers(min_value=3, max_value=5))
    severities = draw(
        st.lists(
            st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    trials = draw(st.lists(st.integers(min_value=10, max_value=40), min_size=n, max_size=n))
    successes = [draw(st.integers(min_value=1, max_value=t - 1)) for t in trials]
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


def _fit_fixed_trivial(obs: Observations) -> Fit:
    return fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)


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
def test_aggregated_and_disaggregated_inputs_give_identical_diagnostics(
    counts: tuple[list[float], list[int], list[int]],
) -> None:
    aggregated = _from_counts(counts)
    disaggregated = _rows_from_counts(counts)

    fit_aggregated = _fit_fixed_trivial(aggregated)
    fit_disaggregated = _fit_fixed_trivial(disaggregated)
    assume(fit_aggregated.status is Status.OK)
    assume(fit_disaggregated.status is Status.OK)

    assert diagnose(fit_aggregated) == diagnose(fit_disaggregated)


@given(_counts_with_optional_control(), st.data())
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_row_permutation_does_not_change_diagnostics(
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

    fit_original = _fit_fixed_trivial(original)
    fit_permuted = _fit_fixed_trivial(permuted)
    assume(fit_original.status is Status.OK)
    assume(fit_permuted.status is Status.OK)

    assert diagnose(fit_original) == diagnose(fit_permuted)


@given(_non_separated_counts(), st.integers(min_value=1, max_value=200))
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_adding_an_all_success_control_leaves_gof_and_dispersion_unchanged(
    counts: tuple[list[float], list[int], list[int]],
    control_trials: int,
) -> None:
    severity, successes, trials = counts

    without_control = _from_counts((severity, successes, trials))
    with_control = _from_counts(
        ([0.0, *severity], [control_trials, *successes], [control_trials, *trials])
    )

    fit_without = _fit_fixed_trivial(without_control)
    fit_with = _fit_fixed_trivial(with_control)
    assume(fit_without.status is Status.OK)
    assume(fit_with.status is Status.OK)

    diag_without = diagnose(fit_without)
    diag_with = diagnose(fit_with)

    # Scope: only the four GOF/dispersion fields, since an added
    # control row adds one more adjacent monotonicity pair and can change which alternative
    # links converge.
    assert diag_with.deviance == diag_without.deviance
    assert diag_with.pearson_chi2 == diag_without.pearson_chi2
    assert diag_with.df == diag_without.df
    assert diag_with.dispersion == diag_without.dispersion
