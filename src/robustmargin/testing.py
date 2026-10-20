"""Schema-valid fake results for **consumers' own tests**.

``fake_fit``, ``fake_threshold`` and ``fake_ratio`` build deterministic, schema-valid
:class:`~robustmargin.Fit`, :class:`~robustmargin.Threshold` and :class:`~robustmargin.Ratio`
objects so a consumer can write and run its own report-rendering and serialisation tests without
fitting any data. They are **not** statistical estimates of anything: every one of them carries
a warning saying so, and nothing here should be read as a claim about what a real fit would
produce for the same inputs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from scipy.stats import binomtest

from robustmargin.empirical import BaselineRate, GridBreakPoint
from robustmargin.models import Covariance, Fit, Parameter
from robustmargin.ratio import Ratio
from robustmargin.threshold import Threshold
from robustmargin.types import (
    Axis,
    Cell,
    Censoring,
    Definition,
    IntervalShape,
    JSONValue,
    Observations,
    Status,
)
from robustmargin.validation import Diagnostics, MonotonicityCheck, check_monotonicity, diagnose

__all__ = ["fake_fit", "fake_ratio", "fake_threshold"]


class _Unset:
    """Sentinel distinguishing "argument not passed" from an explicit ``None`` override for
    ``fake_fit``'s ``diagnostics``/``monotonicity`` parameters: an explicit argument --
    ``None`` included -- always overrides; only *omitting* the argument entirely means "compute
    the real one"."""

    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset()


def _observations_from_cells(fit: Fit) -> Observations:
    """Rebuild the :class:`~robustmargin.Observations` a fake ``fit`` "came from", so
    :func:`~robustmargin.check_monotonicity` can run on it the same way it would on real data."""
    return Observations.from_counts(
        fit.axis,
        severity=[c.severity for c in fit.cells],
        successes=[c.successes for c in fit.cells],
        trials=[c.trials for c in fit.cells],
        outcome=fit.outcome,
        direction=fit.direction,
    )


_AXIS = Axis(name="severity", unit="unit", scale="log")
_FAKE_WARNING = (
    "This is a fake object from robustmargin.testing, built for a consumer's own tests. It is "
    "schema-valid, not a statistical estimate."
)
_CELLS: tuple[Cell, ...] = (
    Cell(severity=0.0, successes=100, trials=100),
    Cell(severity=1.0, successes=60, trials=100),
    Cell(severity=2.0, successes=20, trials=100),
)
# fake_ratio builds its two thresholds from two separate fits on disjoint data (different
# severities entirely), not the same _CELLS reused twice, so a consumer's test never mistakes
# the pair for a ratio computed from one dataset against itself.
_CELLS_A: tuple[Cell, ...] = _CELLS
_CELLS_B: tuple[Cell, ...] = (
    # severity=0.0 is an all-success control, like _CELLS_A's: 'upper' is fixed at exactly 1.0
    # in _fake_ok_params_and_covariance, and a failing control combined with a fixed upper=1.0
    # makes the likelihood zero on a log axis (Fit's own check).
    Cell(severity=0.0, successes=100, trials=100),
    Cell(severity=1.5, successes=50, trials=100),
    Cell(severity=3.0, successes=10, trials=100),
)

# The 14 (status, censoring) combinations a Threshold allows, mirroring the invariants
# Threshold.__post_init__ enforces.
_ALLOWED_THRESHOLD_COMBINATIONS: frozenset[tuple[Status, Censoring]] = frozenset(
    {(Status.OK, censoring) for censoring in Censoring}
    | {
        (status, censoring)
        for status in (Status.SEPARATION, Status.NOT_CONVERGED)
        for censoring in (Censoring.NONE, Censoring.RIGHT, Censoring.LEFT)
    }
    | {
        (status, Censoring.NONE)
        for status in (Status.UNREACHABLE, Status.FAILS_AT_BASELINE, Status.CONTROL_INCOMPATIBLE)
    }
)

# The default fit under a threshold. UNREACHABLE and FAILS_AT_BASELINE are properties of the
# target and the data, not of how the fit went (the target performance is out of the fitted
# curve's range or outside (0, 1), or the control already fails), so a `Threshold` accepts them
# on a fit of any status; the fake builds the common case, a converged fit. SEPARATION,
# NOT_CONVERGED and CONTROL_INCOMPATIBLE are properties of the fit itself, so a threshold with
# one of those statuses is built on a fit carrying the same status.
_FIT_STATUS_FOR_THRESHOLD_STATUS: dict[Status, Status] = {
    Status.OK: Status.OK,
    Status.UNREACHABLE: Status.OK,
    Status.FAILS_AT_BASELINE: Status.OK,
    Status.SEPARATION: Status.SEPARATION,
    Status.NOT_CONVERGED: Status.NOT_CONVERGED,
    Status.CONTROL_INCOMPATIBLE: Status.CONTROL_INCOMPATIBLE,
}


def _fake_provenance(provenance: Mapping[str, JSONValue] | None) -> dict[str, JSONValue]:
    return dict(provenance) if provenance is not None else {}


def _fake_ok_params_and_covariance() -> tuple[dict[str, Parameter], Covariance]:
    params = {
        "mu": Parameter(value=0.5, fixed=False),
        "s": Parameter(value=0.3, fixed=False),
        "upper": Parameter(value=1.0, fixed=True),
        "lower": Parameter(value=0.0, fixed=True),
    }
    covariance = Covariance(names=("mu", "s"), matrix=((0.01, 0.0), (0.0, 0.02)))
    return params, covariance


def _fake_ok_fit(
    cells: tuple[Cell, ...],
    *,
    provenance: Mapping[str, JSONValue] | None = None,
    diagnostics: Diagnostics | None | _Unset = _UNSET,
    monotonicity: MonotonicityCheck | None | _Unset = _UNSET,
) -> Fit:
    params, covariance = _fake_ok_params_and_covariance()
    base = Fit(
        axis=_AXIS,
        outcome="success",
        direction="decreasing",
        model="binomial",
        link="probit",
        params=params,
        covariance=covariance,
        log_likelihood=-12.5,
        cells=cells,
        status=Status.OK,
        cluster_ids=None,
        warnings=(_FAKE_WARNING,),
        provenance=_fake_provenance(provenance),
    )
    # Mirrors a real fit_dose_response result -- diagnostics through diagnose() (status is OK
    # here, so this never raises), monotonicity through check_monotonicity() on the same cells --
    # unless the caller explicitly overrode either.
    resolved_diagnostics = diagnose(base) if isinstance(diagnostics, _Unset) else diagnostics
    resolved_monotonicity = (
        check_monotonicity(_observations_from_cells(base))
        if isinstance(monotonicity, _Unset)
        else monotonicity
    )
    return dataclasses.replace(
        base, diagnostics=resolved_diagnostics, monotonicity=resolved_monotonicity
    )


def fake_fit(
    *,
    status: Status = Status.OK,
    provenance: Mapping[str, JSONValue] | None = None,
    diagnostics: Diagnostics | None | _Unset = _UNSET,
    monotonicity: MonotonicityCheck | None | _Unset = _UNSET,
) -> Fit:
    """Build a deterministic, schema-valid :class:`~robustmargin.Fit`.

    Parameters
    ----------
    status
        The :class:`~robustmargin.Status` to build. Every member other than ``UNREACHABLE`` and
        ``FAILS_AT_BASELINE`` is accepted (those describe a threshold, not a fit, and
        :class:`~robustmargin.Fit` itself rejects them). When ``status`` is not ``OK``, ``params``,
        ``covariance`` and ``log_likelihood`` are all ``None``, matching
        :class:`~robustmargin.Fit`'s own invariant.
    provenance
        Stored on the result as given, or ``{}`` if not provided.
    diagnostics
        Mirrors a real fit: when omitted entirely, this is :func:`~robustmargin.diagnose` on the
        built fit when ``status`` is ``OK``, or ``None`` otherwise (:func:`~robustmargin.diagnose`
        itself requires ``status is Status.OK``). Passing an explicit value -- ``None`` included --
        always overrides that and is stored as given; ``Fit`` itself still rejects a non-``None``
        value when ``status`` is not ``OK`` (a failed fit never carries a number).
    monotonicity
        Mirrors a real fit, the same way ``diagnostics`` does: when omitted entirely, this is
        :func:`~robustmargin.check_monotonicity` on the built fit's own cells, for *every*
        ``status`` (a real :func:`~robustmargin.fit_dose_response` result is never ``None`` here
        regardless of status). An explicit value -- ``None`` included -- always overrides.

    Returns
    -------
    Fit
        A binomial-probit fit on a neutral ``severity`` axis, with ``cluster_ids=None`` and
        ``warnings`` naming it a fake.
    """
    if status is Status.OK:
        return _fake_ok_fit(
            _CELLS, provenance=provenance, diagnostics=diagnostics, monotonicity=monotonicity
        )

    base = Fit(
        axis=_AXIS,
        outcome="success",
        direction="decreasing",
        model="binomial",
        link="probit",
        params=None,
        covariance=None,
        log_likelihood=None,
        cells=_CELLS,
        status=status,
        cluster_ids=None,
        warnings=(_FAKE_WARNING,),
        provenance=_fake_provenance(provenance),
    )
    resolved_diagnostics = None if isinstance(diagnostics, _Unset) else diagnostics
    resolved_monotonicity = (
        check_monotonicity(_observations_from_cells(base))
        if isinstance(monotonicity, _Unset)
        else monotonicity
    )
    return dataclasses.replace(
        base, diagnostics=resolved_diagnostics, monotonicity=resolved_monotonicity
    )


def fake_threshold(
    *,
    status: Status = Status.OK,
    censoring: Censoring = Censoring.NONE,
    provenance: Mapping[str, JSONValue] | None = None,
) -> Threshold:
    """Build a deterministic, schema-valid :class:`~robustmargin.Threshold`.

    Parameters
    ----------
    status
        The :class:`~robustmargin.Status` to build.
    censoring
        The :class:`~robustmargin.Censoring` to build.
    provenance
        Stored on the result as given, or ``{}`` if not provided.

    Returns
    -------
    Threshold
        A threshold on a neutral ``severity`` axis with ``dependence="independent"``, and
        ``warnings`` naming it a fake.

    Raises
    ------
    ValueError
        If ``(status, censoring)`` is not one of the fourteen combinations a ``Threshold``
        allows.
    """
    if (status, censoring) not in _ALLOWED_THRESHOLD_COMBINATIONS:
        raise ValueError(
            f"fake_threshold: status={status!r} with censoring={censoring!r} is not a "
            "combination Threshold allows"
        )

    fit = fake_fit(status=_FIT_STATUS_FOR_THRESHOLD_STATUS[status])

    value: float | None
    lo: float | None
    hi: float | None
    interval_method: str

    if status in (Status.UNREACHABLE, Status.FAILS_AT_BASELINE, Status.CONTROL_INCOMPATIBLE):
        value, lo, hi = None, None, None
        interval_method = "exact_bound"
    elif censoring is Censoring.RIGHT:
        value, lo, hi = None, 10.0, None
        interval_method = "exact_bound"
    elif censoring is Censoring.LEFT:
        value, lo, hi = None, None, 0.01
        interval_method = "exact_bound"
    elif censoring is Censoring.OPEN_UPPER:
        value, lo, hi = 0.05, 0.03, None
        interval_method = "profile"
    elif censoring is Censoring.OPEN_LOWER:
        value, lo, hi = 0.05, None, 0.08
        interval_method = "profile"
    elif status is Status.OK:
        value, lo, hi = 0.05, 0.03, 0.08
        interval_method = "profile"
    else:
        # SEPARATION / NOT_CONVERGED with censoring NONE (the exact-bound fallback, with both
        # sides happening to be available here).
        value, lo, hi = None, 0.02, 0.09
        interval_method = "exact_bound"

    # The baseline and grid fields are set exactly where the invariants allow them, so a
    # consumer testing against a fake sees the same shape a real result has. A fake that left
    # them None on FAILS_AT_BASELINE would quietly stop exercising the field that status exists
    # to carry.
    baseline: BaselineRate | None = None
    grid: GridBreakPoint | None = None
    if status is Status.FAILS_AT_BASELINE:
        # The two-sided exact interval for 98/200 at level=0.95, which is what
        # `censoring._baseline_rate` produces for the same counts. Computed rather than
        # transcribed, so the fake cannot drift from the real path.
        _lo, _hi = binomtest(98, 200).proportion_ci(confidence_level=0.95, method="exact")
        baseline = BaselineRate(
            severity=0.0,
            successes=98,
            trials=200,
            rate=0.49,
            lo=float(_lo),
            hi=float(_hi),
            level=0.95,
        )
    elif status in (Status.SEPARATION, Status.NOT_CONVERGED):
        grid = GridBreakPoint(0.05, 0.1, Censoring.NONE)

    return Threshold(
        axis=_AXIS,
        definition=Definition.absolute(0.95),
        direction="decreasing",
        value=value,
        lo=lo,
        hi=hi,
        level=0.95,
        interval_method=interval_method,
        censoring=censoring,
        status=status,
        dependence="independent",
        fit=fit,
        warnings=(_FAKE_WARNING,),
        baseline=baseline,
        grid=grid,
        provenance=_fake_provenance(provenance),
    )


def _fake_ok_threshold(*, fit: Fit, value: float, lo: float, hi: float) -> Threshold:
    return Threshold(
        axis=_AXIS,
        definition=Definition.absolute(0.95),
        direction="decreasing",
        value=value,
        lo=lo,
        hi=hi,
        level=0.95,
        interval_method="profile",
        censoring=Censoring.NONE,
        status=Status.OK,
        dependence="independent",
        fit=fit,
        warnings=(_FAKE_WARNING,),
        provenance={},
    )


def fake_ratio(
    *,
    shape: IntervalShape | None = None,
    status: Status = Status.OK,
    censoring: Censoring = Censoring.NONE,
    provenance: Mapping[str, JSONValue] | None = None,
) -> Ratio:
    """Build a deterministic, schema-valid :class:`~robustmargin.Ratio`.

    A failed or censored ratio carries no numbers: whenever ``status`` is not
    ``OK`` or ``censoring`` is not ``NONE``, ``shape``, ``estimate``, ``lo`` and ``hi`` are all
    ``None``, matching :class:`~robustmargin.Ratio`'s own invariant.

    Parameters
    ----------
    shape
        The :class:`~robustmargin.IntervalShape` to build. ``None`` (the default) means
        ``BOUNDED`` when ``status`` is ``OK`` and ``censoring`` is ``NONE``, and means "no
        shape" (forced ``None``) otherwise. Passing an explicit shape together with a non-``OK``
        ``status`` or a non-``NONE`` ``censoring`` raises ``ValueError``, since that combination
        can never be a valid ``Ratio``. Every member is buildable, including
        ``IntervalShape.EXCLUSIVE``, which a real :func:`~robustmargin.ratio_interval` call cannot
        produce while ``dependence="independent"`` but whose :class:`~robustmargin.Ratio` invariants
        still need a consumer-testable fixture.
    status
        The :class:`~robustmargin.Status` to build.
    censoring
        The :class:`~robustmargin.Censoring` to build.
    provenance
        Stored on the result as given, or ``{}`` if not provided.

    Returns
    -------
    Ratio
        A ratio built from two fits on disjoint data (different cells entirely), with
        ``dependence="independent"``, ``cluster_ids=None`` on both fits, and ``warnings``
        naming it a fake. On the ``OK`` + ``NONE`` path, ``estimate`` is exactly
        ``threshold_a.value / threshold_b.value``, matching
        :class:`~robustmargin.Ratio`'s own invariant; the point estimate always lies inside the
        Fieller confidence set described by ``shape``, including for ``EXCLUSIVE``.

    Raises
    ------
    ValueError
        If ``shape`` is given explicitly while ``status`` is not ``OK`` or ``censoring`` is not
        ``NONE``.
    """
    is_ok_none = status is Status.OK and censoring is Censoring.NONE
    if not is_ok_none and shape is not None:
        raise ValueError(
            f"fake_ratio: shape={shape!r} is not allowed when status={status!r} or "
            f"censoring={censoring!r} -- a failed or censored ratio carries no shape"
        )

    value_a, lo_a, hi_a = 0.05, 0.03, 0.08
    value_b, lo_b, hi_b = 0.1, 0.06, 0.16
    threshold_a = _fake_ok_threshold(fit=_fake_ok_fit(_CELLS_A), value=value_a, lo=lo_a, hi=hi_a)
    threshold_b = _fake_ok_threshold(fit=_fake_ok_fit(_CELLS_B), value=value_b, lo=lo_b, hi=hi_b)

    resolved_shape: IntervalShape | None
    estimate: float | None
    lo: float | None
    hi: float | None

    if not is_ok_none:
        resolved_shape, estimate, lo, hi = None, None, None, None
    else:
        resolved_shape = IntervalShape.BOUNDED if shape is None else shape
        ratio_value = value_a / value_b
        estimate = ratio_value
        if resolved_shape is IntervalShape.BOUNDED:
            lo, hi = 0.3, 0.7
        elif resolved_shape is IntervalShape.UNBOUNDED:
            lo, hi = None, None
        elif resolved_shape is IntervalShape.HALF_OPEN:
            # [lo, inf): lo <= estimate is Ratio's own invariant. This is a shape a real
            # ratio_interval(method="fieller") call does produce -- it covers an open region of
            # the parameter space -- unlike EXCLUSIVE below.
            lo, hi = ratio_value * 0.5, None
        else:
            # EXCLUSIVE: unreachable through ratio_interval() itself while
            # dependence="independent" fixes the covariance term at 0, but Ratio's own
            # invariants for it stay enforced against a future dependence="paired", and this fake
            # builds one directly so a consumer can still test against it. The point estimate
            # sits on the near edge of the excluded gap, inside the (-inf, lo] ray of the
            # Fieller confidence set.
            lo, hi = ratio_value, ratio_value + 1.0

    return Ratio(
        threshold_a=threshold_a,
        threshold_b=threshold_b,
        estimate=estimate,
        lo=lo,
        hi=hi,
        shape=resolved_shape,
        method="fieller",
        dependence="independent",
        level=0.95,
        censoring=censoring,
        status=status,
        warnings=(_FAKE_WARNING,),
        provenance=_fake_provenance(provenance),
    )
