"""Schema-valid fake results for **consumers' own tests**.

``fake_fit``, ``fake_threshold`` and ``fake_ratio`` build deterministic, schema-valid
:class:`~robustmargin.Fit`, :class:`~robustmargin.Threshold` and :class:`~robustmargin.Ratio`
objects so a consumer can write and run its own report-rendering and serialisation tests without
fitting any data. They are **not** statistical estimates of anything: every one of them carries
a warning saying so, and nothing here should be read as a claim about what a real fit would
produce for the same inputs.

``fake_fit`` and ``fake_threshold`` build a binomial result by default, a Gaussian
(continuous-response) one with ``model="gaussian"`` and an isotonic one with ``model="isotonic"``.
``fake_threshold`` can also build a bootstrap-interval result (``interval_method``). With
``dependence="clustered"`` ``fake_fit`` and ``fake_threshold`` build a clustered binomial result
(a cluster-robust covariance, ``cluster_cells`` and a cluster sign test), and ``fake_ratio`` with
``dependence="paired"`` builds a ratio of two such results on one set of clusters.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

from scipy.stats import binomtest
from scipy.stats import f as f_distribution

from robustmargin.empirical import BaselineRate, GridBreakPoint
from robustmargin.models import (
    Covariance,
    Fit,
    Parameter,
    _count_cells,
    _few_clusters_warning,
    _isotonic_estimates,
    _isotonic_log_likelihood,
)
from robustmargin.ratio import Ratio
from robustmargin.threshold import Threshold
from robustmargin.types import (
    Axis,
    BootstrapSummary,
    Cell,
    Censoring,
    ClusterCell,
    ContinuousCell,
    ContinuousObservations,
    Definition,
    IntervalShape,
    JSONValue,
    Observations,
    Residual,
    Status,
)
from robustmargin.validation import (
    ContinuousDiagnostics,
    Diagnostics,
    MonotonicityCheck,
    check_monotonicity,
    diagnose,
)

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
    :func:`~robustmargin.check_monotonicity` can run on it the same way it would on real data. A
    clustered fake's observations carry its cluster ids, one row per ``cluster_cells`` entry."""
    if fit.cluster_cells is not None:
        return Observations.from_counts(
            fit.axis,
            severity=[c.severity for c in fit.cluster_cells],
            successes=[c.successes for c in fit.cluster_cells],
            trials=[c.trials for c in fit.cluster_cells],
            outcome=fit.outcome,
            direction=fit.direction,
            cluster=[c.cluster for c in fit.cluster_cells],
        )
    cells = _count_cells(fit)
    return Observations.from_counts(
        fit.axis,
        severity=[c.severity for c in cells],
        successes=[c.successes for c in cells],
        trials=[c.trials for c in cells],
        outcome=fit.outcome,
        direction=fit.direction,
    )


def _fake_monotonicity(fit: Fit) -> MonotonicityCheck:
    """The monotonicity check a real fit of the same data would carry: the cluster sign test for
    a clustered fake, Fisher's test otherwise."""
    obs = _observations_from_cells(fit)
    if fit.dependence == "clustered":
        return check_monotonicity(obs, dependence="clustered")
    return check_monotonicity(obs)


_FAKE_CLUSTERS = 4
_VALID_DEPENDENCE = frozenset({"independent", "clustered"})
_VALID_RATIO_DEPENDENCE = frozenset({"independent", "paired"})


def _fake_cluster_cells(cells: tuple[Cell, ...]) -> tuple[tuple[str, ...], tuple[ClusterCell, ...]]:
    """Split each cell's counts across ``_FAKE_CLUSTERS`` clusters of equal trials, so the cluster
    cells sum to ``cells`` exactly. An interior count is spread unevenly (``+1, -1, +1, -1`` about
    an even split) so the clusters are not identical."""
    ids = tuple(f"cluster_{i}" for i in range(_FAKE_CLUSTERS))
    cluster_cells: list[ClusterCell] = []
    for cell in cells:
        trials = cell.trials // _FAKE_CLUSTERS
        base, extra = divmod(cell.successes, _FAKE_CLUSTERS)
        for index, cluster in enumerate(ids):
            successes = base + (1 if index < extra else 0)
            if 0 < cell.successes < cell.trials and extra == 0:
                successes += 1 if index % 2 == 0 else -1
            cluster_cells.append(
                ClusterCell(
                    severity=cell.severity, cluster=cluster, successes=successes, trials=trials
                )
            )
    return ids, tuple(cluster_cells)


def _fake_cluster_fields(
    cells: tuple[Cell, ...], dependence: str
) -> tuple[tuple[str, ...] | None, tuple[ClusterCell, ...] | None, tuple[str, ...]]:
    """``(cluster_ids, cluster_cells, warnings)`` a fake of the given ``dependence`` carries:
    nothing for an independent one, four clusters and a few-clusters note for a clustered one."""
    if dependence != "clustered":
        return None, None, ()
    ids, cluster_cells = _fake_cluster_cells(cells)
    return ids, cluster_cells, (_few_clusters_warning(len(ids)),)


def _check_dependence(dependence: str, *, function: str, valid: frozenset[str]) -> None:
    if dependence not in valid:
        raise ValueError(
            f"{function}: dependence must be one of {sorted(valid)}, got {dependence!r}"
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
# The increasing counterpart: events rise with severity from a control with none, so a lower
# asymptote fixed at exactly 0.0 is compatible with it.
_CELLS_INCREASING: tuple[Cell, ...] = (
    Cell(severity=0.0, successes=0, trials=100),
    Cell(severity=1.0, successes=40, trials=100),
    Cell(severity=2.0, successes=80, trials=100),
)
_VALID_DIRECTIONS = frozenset({"decreasing", "increasing"})
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

# A Gaussian threshold makes no exact claim.
_ALLOWED_GAUSSIAN_THRESHOLD_COMBINATIONS: frozenset[tuple[Status, Censoring]] = frozenset(
    {
        (Status.OK, Censoring.NONE),
        (Status.OK, Censoring.OPEN_UPPER),
        (Status.OK, Censoring.OPEN_LOWER),
        (Status.UNREACHABLE, Censoring.NONE),
        (Status.NOT_CONVERGED, Censoring.NONE),
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


def _check_model(model: str, *, function: str, allowed: frozenset[str] | None = None) -> None:
    valid = _VALID_MODELS if allowed is None else allowed
    if model not in valid:
        raise ValueError(f"{function}: model must be one of {sorted(valid)}, got {model!r}")


def _check_direction(direction: str, *, function: str) -> None:
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"{function}: direction must be one of {sorted(_VALID_DIRECTIONS)}, got {direction!r}"
        )


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
    direction: str = "decreasing",
    provenance: Mapping[str, JSONValue] | None = None,
    diagnostics: Diagnostics | None | _Unset = _UNSET,
    monotonicity: MonotonicityCheck | None | _Unset = _UNSET,
    dependence: str = "independent",
) -> Fit:
    params, covariance = _fake_ok_params_and_covariance()
    cluster_ids, cluster_cells, cluster_warnings = _fake_cluster_fields(cells, dependence)
    base = Fit(
        axis=_AXIS,
        outcome="success",
        direction=direction,
        model="binomial",
        link="probit",
        params=params,
        covariance=covariance,
        log_likelihood=-12.5,
        cells=cells,
        status=Status.OK,
        cluster_ids=cluster_ids,
        warnings=(_FAKE_WARNING, *cluster_warnings),
        provenance=_fake_provenance(provenance),
        dependence=dependence,
        cluster_cells=cluster_cells,
    )
    # Mirrors a real fit_dose_response result -- diagnostics through diagnose() (status is OK
    # here, so this never raises), monotonicity through check_monotonicity() on the same cells --
    # unless the caller explicitly overrode either.
    resolved_diagnostics = diagnose(base) if isinstance(diagnostics, _Unset) else diagnostics
    resolved_monotonicity = (
        _fake_monotonicity(base) if isinstance(monotonicity, _Unset) else monotonicity
    )
    return dataclasses.replace(
        base, diagnostics=resolved_diagnostics, monotonicity=resolved_monotonicity
    )


_VALID_MODELS = frozenset({"binomial", "gaussian", "isotonic"})
_MODEL_INTERVAL_METHODS = frozenset({"profile", "delta"})
_INDEPENDENT_BOOTSTRAP_METHODS = frozenset({"parametric_bootstrap", "cell_bootstrap"})
_BOOTSTRAP_INTERVAL_METHODS = _INDEPENDENT_BOOTSTRAP_METHODS | {"cluster_bootstrap"}
_CLUSTERED_INTERVAL_METHODS = frozenset({"cluster_bootstrap", "delta"})
# A clustered threshold makes no exact claim: neither RIGHT, LEFT nor FAILS_AT_BASELINE, and a
# failed fit reports no bound.
_ALLOWED_CLUSTERED_THRESHOLD_COMBINATIONS: frozenset[tuple[Status, Censoring]] = frozenset(
    {
        (Status.OK, Censoring.NONE),
        (Status.OK, Censoring.OPEN_UPPER),
        (Status.OK, Censoring.OPEN_LOWER),
        (Status.UNREACHABLE, Censoring.NONE),
        (Status.SEPARATION, Censoring.NONE),
        (Status.NOT_CONVERGED, Censoring.NONE),
        (Status.CONTROL_INCOMPATIBLE, Censoring.NONE),
    }
)
# A Gaussian fake: a decreasing logit curve from 10 to 2 with its midpoint at severity 1, three
# observations per level with a sample SD of 0.3. The increasing one rises from 2 to 10.
_GAUSSIAN_MEANS_DECREASING = (10.0, 8.4, 6.0, 3.6, 2.5)
_GAUSSIAN_MEANS_INCREASING = (2.0, 3.6, 6.0, 8.4, 9.5)
_GAUSSIAN_SEVERITY = (0.0, 0.5, 1.0, 2.0, 4.0)
_GAUSSIAN_RSS = 1.0
_GAUSSIAN_PURE_ERROR_SS = 5 * 2 * 0.3 * 0.3


def _gaussian_cells(direction: str) -> tuple[ContinuousCell, ...]:
    means = _GAUSSIAN_MEANS_DECREASING if direction == "decreasing" else _GAUSSIAN_MEANS_INCREASING
    return tuple(
        ContinuousCell(severity=severity, n=3, mean=mean, sd=0.3)
        for severity, mean in zip(_GAUSSIAN_SEVERITY, means, strict=True)
    )


def _gaussian_observations_from_cells(fit: Fit) -> ContinuousObservations:
    """The :class:`~robustmargin.ContinuousObservations` a fake Gaussian ``fit`` "came from"."""
    cells = [c for c in fit.cells if isinstance(c, ContinuousCell)]
    return ContinuousObservations.from_summaries(
        fit.axis,
        severity=[c.severity for c in cells],
        mean=[c.mean for c in cells],
        sd=[c.sd for c in cells],
        n=[c.n for c in cells],
        outcome=fit.outcome,
        direction=fit.direction,
    )


def _gaussian_fake_diagnostics() -> ContinuousDiagnostics:
    pure_error_df = 10
    lack_of_fit_df = 1
    statistic = ((_GAUSSIAN_RSS - _GAUSSIAN_PURE_ERROR_SS) / lack_of_fit_df) / (
        _GAUSSIAN_PURE_ERROR_SS / pure_error_df
    )
    return ContinuousDiagnostics(
        rss=_GAUSSIAN_RSS,
        df=11,
        pure_error_ss=_GAUSSIAN_PURE_ERROR_SS,
        pure_error_df=pure_error_df,
        lack_of_fit_f=statistic,
        lack_of_fit_p=float(f_distribution.sf(statistic, lack_of_fit_df, pure_error_df)),
        link_aic={"probit": 21.0, "logit": 20.0, "cloglog": 24.0},
        link_status={"probit": Status.OK, "logit": Status.OK, "cloglog": Status.OK},
    )


def _gaussian_fake_fit(
    *,
    status: Status,
    direction: str,
    provenance: Mapping[str, JSONValue] | None,
    diagnostics: Diagnostics | ContinuousDiagnostics | None | _Unset,
    monotonicity: MonotonicityCheck | None | _Unset,
) -> Fit:
    if status not in (Status.OK, Status.NOT_CONVERGED):
        raise ValueError(
            f"fake_fit: status={status!r} is not one a Gaussian fit has; the statuses are OK "
            "and NOT_CONVERGED"
        )
    cells = _gaussian_cells(direction)
    ok = status is Status.OK
    params = (
        {
            "mu": Parameter(value=0.0, fixed=False),
            "s": Parameter(value=0.5, fixed=False),
            "upper": Parameter(value=10.0, fixed=False),
            "lower": Parameter(value=2.0, fixed=False),
        }
        if ok
        else None
    )
    covariance = (
        Covariance(
            names=("mu", "s", "upper", "lower"),
            matrix=(
                (0.01, 0.0, 0.0, 0.0),
                (0.0, 0.005, 0.0, 0.0),
                (0.0, 0.0, 0.04, 0.0),
                (0.0, 0.0, 0.0, 0.03),
            ),
        )
        if ok
        else None
    )
    base = Fit(
        axis=_AXIS,
        outcome="response",
        direction=direction,
        model="gaussian",
        link="logit",
        params=params,
        covariance=covariance,
        log_likelihood=-0.5 * 15 * (math.log(2.0 * math.pi * _GAUSSIAN_RSS / 15) + 1.0)
        if ok
        else None,
        cells=cells,
        status=status,
        cluster_ids=None,
        warnings=(_FAKE_WARNING,),
        provenance=_fake_provenance(provenance),
        residual=Residual(sigma=math.sqrt(_GAUSSIAN_RSS / 11), df=11, rss=_GAUSSIAN_RSS)
        if ok
        else None,
    )
    resolved_diagnostics = (
        (_gaussian_fake_diagnostics() if ok else None)
        if isinstance(diagnostics, _Unset)
        else diagnostics
    )
    resolved_monotonicity = (
        check_monotonicity(_gaussian_observations_from_cells(base))
        if isinstance(monotonicity, _Unset)
        else monotonicity
    )
    return dataclasses.replace(
        base, diagnostics=resolved_diagnostics, monotonicity=resolved_monotonicity
    )


def _isotonic_fake_fit(
    *,
    status: Status,
    direction: str,
    provenance: Mapping[str, JSONValue] | None,
    diagnostics: Diagnostics | ContinuousDiagnostics | None | _Unset,
    monotonicity: MonotonicityCheck | None | _Unset,
) -> Fit:
    """An isotonic fake: the binomial fake's counts, fitted by pool-adjacent-violators."""
    if status is not Status.OK:
        raise ValueError(
            f"fake_fit: status={status!r} is not one an isotonic fit has; it is always OK"
        )
    if not isinstance(diagnostics, _Unset) and diagnostics is not None:
        raise ValueError("fake_fit: an isotonic fit has no diagnostics")
    cells = _CELLS if direction == "decreasing" else _CELLS_INCREASING
    estimates = _isotonic_estimates(cells, direction=direction)
    base = Fit(
        axis=_AXIS,
        outcome="success",
        direction=direction,
        model="isotonic",
        link="none",
        params=None,
        covariance=None,
        log_likelihood=_isotonic_log_likelihood(cells, estimates),
        cells=cells,
        status=Status.OK,
        cluster_ids=None,
        warnings=(_FAKE_WARNING,),
        provenance=_fake_provenance(provenance),
        cell_estimates=estimates,
    )
    resolved_monotonicity = (
        check_monotonicity(_observations_from_cells(base))
        if isinstance(monotonicity, _Unset)
        else monotonicity
    )
    return dataclasses.replace(base, monotonicity=resolved_monotonicity)


def fake_fit(
    *,
    model: str = "binomial",
    status: Status = Status.OK,
    direction: str = "decreasing",
    provenance: Mapping[str, JSONValue] | None = None,
    diagnostics: Diagnostics | ContinuousDiagnostics | None | _Unset = _UNSET,
    monotonicity: MonotonicityCheck | None | _Unset = _UNSET,
    dependence: str = "independent",
) -> Fit:
    """Build a deterministic, schema-valid :class:`~robustmargin.Fit`.

    Parameters
    ----------
    model
        ``"binomial"`` (the default), ``"gaussian"`` or ``"isotonic"``. A Gaussian fake is a logit
        fit on a neutral ``severity`` axis with five levels of three observations, ``residual``,
        :class:`~robustmargin.ContinuousDiagnostics` and a Welch monotonicity check. It accepts
        only ``status`` ``OK`` and ``NOT_CONVERGED``, the two a Gaussian fit has. An isotonic fake
        is the binomial fake's three levels of counts with their ``cell_estimates``, and accepts
        only ``status`` ``OK``, the one an isotonic fit has.
    status
        The :class:`~robustmargin.Status` to build. Every member other than ``UNREACHABLE`` and
        ``FAILS_AT_BASELINE`` is accepted (those describe a threshold, not a fit, and
        :class:`~robustmargin.Fit` itself rejects them). When ``status`` is not ``OK``, ``params``,
        ``covariance`` and ``log_likelihood`` are all ``None``, matching
        :class:`~robustmargin.Fit`'s own invariant.
    direction
        ``"decreasing"`` (the default) or ``"increasing"``. An increasing fit counts an outcome
        that rises with severity, from a control with none to most trials at the highest
        severity.
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
    dependence
        ``"independent"`` (the default) or ``"clustered"``. A clustered fake is a binomial fit
        with four clusters: ``cluster_ids``, ``cluster_cells`` summing to ``cells``, a
        ``"FEW_CLUSTERS: "`` warning, a cluster sign test as its monotonicity check and
        diagnostics carrying the ``"CLUSTERED_GOF: "`` note.

    Returns
    -------
    Fit
        A binomial-probit fit on a neutral ``severity`` axis, with ``cluster_ids=None`` unless
        ``dependence="clustered"``, and ``warnings`` naming it a fake.

    Raises
    ------
    ValueError
        If ``model`` is not ``"binomial"``, ``"gaussian"`` or ``"isotonic"``, ``direction`` is
        not ``"decreasing"`` or ``"increasing"``, ``dependence`` is not ``"independent"`` or
        ``"clustered"``, ``dependence`` is ``"clustered"`` and ``model`` is not ``"binomial"``,
        ``model`` is ``"gaussian"`` and ``status`` is not ``OK`` or ``NOT_CONVERGED``, or
        ``model`` is ``"isotonic"`` and ``status`` is not ``OK`` (or ``diagnostics`` is given).
    """
    _check_model(model, function="fake_fit")
    _check_direction(direction, function="fake_fit")
    _check_dependence(dependence, function="fake_fit", valid=_VALID_DEPENDENCE)
    if dependence == "clustered" and model != "binomial":
        raise ValueError(
            f"fake_fit: dependence='clustered' needs model='binomial', got {model!r}: clustered "
            "continuous inference is not implemented"
        )
    if model == "isotonic":
        return _isotonic_fake_fit(
            status=status,
            direction=direction,
            provenance=provenance,
            diagnostics=diagnostics,
            monotonicity=monotonicity,
        )
    if model == "gaussian":
        return _gaussian_fake_fit(
            status=status,
            direction=direction,
            provenance=provenance,
            diagnostics=diagnostics,
            monotonicity=monotonicity,
        )
    if isinstance(diagnostics, ContinuousDiagnostics):
        raise ValueError("fake_fit: ContinuousDiagnostics belong to model='gaussian'")
    cells = _CELLS if direction == "decreasing" else _CELLS_INCREASING
    if status is Status.OK:
        return _fake_ok_fit(
            cells,
            direction=direction,
            provenance=provenance,
            diagnostics=diagnostics,
            monotonicity=monotonicity,
            dependence=dependence,
        )

    cluster_ids, cluster_cells, cluster_warnings = _fake_cluster_fields(cells, dependence)
    base = Fit(
        axis=_AXIS,
        outcome="success",
        direction=direction,
        model="binomial",
        link="probit",
        params=None,
        covariance=None,
        log_likelihood=None,
        cells=cells,
        status=status,
        cluster_ids=cluster_ids,
        warnings=(_FAKE_WARNING, *cluster_warnings),
        provenance=_fake_provenance(provenance),
        dependence=dependence,
        cluster_cells=cluster_cells,
    )
    resolved_diagnostics = None if isinstance(diagnostics, _Unset) else diagnostics
    resolved_monotonicity = (
        _fake_monotonicity(base) if isinstance(monotonicity, _Unset) else monotonicity
    )
    return dataclasses.replace(
        base, diagnostics=resolved_diagnostics, monotonicity=resolved_monotonicity
    )


def fake_threshold(
    *,
    model: str = "binomial",
    status: Status = Status.OK,
    censoring: Censoring = Censoring.NONE,
    direction: str = "decreasing",
    provenance: Mapping[str, JSONValue] | None = None,
    interval_method: str | None = None,
    dependence: str = "independent",
) -> Threshold:
    """Build a deterministic, schema-valid :class:`~robustmargin.Threshold`.

    Parameters
    ----------
    model
        ``"binomial"`` (the default), ``"gaussian"`` or ``"isotonic"``. A Gaussian threshold is on
        a Gaussian fake fit (see :func:`fake_fit`) and makes no exact claim: only ``(OK, NONE)``,
        ``(OK, OPEN_UPPER)``, ``(OK, OPEN_LOWER)``, ``(UNREACHABLE, NONE)`` and
        ``(NOT_CONVERGED, NONE)`` are buildable, the last two with no value and no bound. An
        isotonic threshold is on an isotonic fake fit; its interval is a ``"cell_bootstrap"`` one
        with a :class:`~robustmargin.BootstrapSummary`, and only the statuses of a fit that
        cannot fail (``OK``, ``UNREACHABLE``, ``FAILS_AT_BASELINE``) are buildable.
    status
        The :class:`~robustmargin.Status` to build.
    censoring
        The :class:`~robustmargin.Censoring` to build.
    direction
        ``"decreasing"`` (the default) or ``"increasing"``; the threshold's fit has the same
        direction (see :func:`fake_fit`).
    provenance
        Stored on the result as given, or ``{}`` if not provided.
    interval_method
        ``None`` (the default) builds what the other arguments imply: a ``"profile"`` interval, or
        ``"cell_bootstrap"`` for an isotonic fake. ``"profile"`` and ``"delta"`` build an ``OK``
        threshold with that method and no summary; ``"parametric_bootstrap"`` or
        ``"cell_bootstrap"`` builds one with that method and a
        :class:`~robustmargin.BootstrapSummary` (``censoring`` ``NONE``, ``OPEN_UPPER`` or
        ``OPEN_LOWER`` in every case). A Gaussian fake does not take ``"cell_bootstrap"`` and an
        isotonic one takes only ``"cell_bootstrap"``. With ``dependence="clustered"`` the default
        is ``"cluster_bootstrap"`` and ``"delta"`` is the only other method.
    dependence
        ``"independent"`` (the default) or ``"clustered"``, a binomial threshold on a clustered
        fake fit (see :func:`fake_fit`). A clustered threshold makes no exact claim, so
        ``FAILS_AT_BASELINE`` and ``RIGHT`` and ``LEFT`` censoring are not buildable, and a
        failed fit carries no bound.

    Returns
    -------
    Threshold
        A threshold on a neutral ``severity`` axis with the given ``dependence``, and
        ``warnings`` naming it a fake.

    Raises
    ------
    ValueError
        If ``model`` is not ``"binomial"``, ``"gaussian"`` or ``"isotonic"``, ``(status,
        censoring)`` is not one of the fourteen combinations a ``Threshold`` allows (or one of the
        five a Gaussian one does, one an isotonic fit can have, or one of the seven a clustered
        one does), ``direction`` is not ``"decreasing"`` or ``"increasing"``, ``dependence`` is
        not ``"independent"`` or ``"clustered"`` (or is ``"clustered"`` with a non-binomial
        ``model``), or ``interval_method`` is not one of the above for this ``model``,
        ``dependence``, ``status`` and ``censoring``.
    """
    _check_model(model, function="fake_threshold")
    _check_direction(direction, function="fake_threshold")
    _check_dependence(dependence, function="fake_threshold", valid=_VALID_DEPENDENCE)
    clustered = dependence == "clustered"
    if clustered and model != "binomial":
        raise ValueError(
            f"fake_threshold: dependence='clustered' needs model='binomial', got {model!r}: "
            "clustered continuous inference is not implemented"
        )
    interval_method = _resolve_fake_interval_method(
        interval_method, model=model, status=status, censoring=censoring, clustered=clustered
    )
    allowed = (
        _ALLOWED_CLUSTERED_THRESHOLD_COMBINATIONS
        if clustered
        else (_ALLOWED_THRESHOLD_COMBINATIONS)
    )
    if (status, censoring) not in allowed or (
        model == "gaussian" and (status, censoring) not in _ALLOWED_GAUSSIAN_THRESHOLD_COMBINATIONS
    ):
        raise ValueError(
            f"fake_threshold: status={status!r} with censoring={censoring!r} is not a "
            f"combination a {dependence} {model} Threshold allows"
        )

    fit = fake_fit(
        model=model,
        status=_FIT_STATUS_FOR_THRESHOLD_STATUS[status],
        direction=direction,
        dependence=dependence,
    )
    if model == "gaussian":
        return _gaussian_fake_threshold(
            fit=fit,
            status=status,
            censoring=censoring,
            provenance=provenance,
            interval_method=interval_method,
        )

    value: float | None
    lo: float | None
    hi: float | None
    method: str

    if status in (Status.UNREACHABLE, Status.FAILS_AT_BASELINE, Status.CONTROL_INCOMPATIBLE):
        value, lo, hi = None, None, None
        method = "exact_bound"
    elif clustered and status is not Status.OK:
        # No exact-bound fallback under clustering: a failed fit reports no number at all.
        value, lo, hi = None, None, None
        method = "exact_bound"
    elif censoring is Censoring.RIGHT:
        value, lo, hi = None, 10.0, None
        method = "exact_bound"
    elif censoring is Censoring.LEFT:
        value, lo, hi = None, None, 0.01
        method = "exact_bound"
    elif censoring is Censoring.OPEN_UPPER:
        value, lo, hi = 0.05, 0.03, None
        method = interval_method
    elif censoring is Censoring.OPEN_LOWER:
        value, lo, hi = 0.05, None, 0.08
        method = interval_method
    elif status is Status.OK:
        value, lo, hi = 0.05, 0.03, 0.08
        method = interval_method
    else:
        # SEPARATION / NOT_CONVERGED with censoring NONE (the exact-bound fallback, with both
        # sides happening to be available here).
        value, lo, hi = None, 0.02, 0.09
        method = "exact_bound"
    bootstrap = _fake_bootstrap(method, censoring)

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
    elif status in (Status.SEPARATION, Status.NOT_CONVERGED) and not clustered:
        grid = GridBreakPoint(0.05, 0.1, Censoring.NONE)

    return Threshold(
        axis=_AXIS,
        definition=Definition.absolute(0.95 if direction == "decreasing" else 0.05),
        direction=direction,
        value=value,
        lo=lo,
        hi=hi,
        level=0.95,
        interval_method=method,
        censoring=censoring,
        status=status,
        dependence=dependence,
        fit=fit,
        warnings=(_FAKE_WARNING, *_fake_cluster_warnings(fit, method)),
        baseline=baseline,
        grid=grid,
        provenance=_fake_provenance(provenance),
        bootstrap=bootstrap,
    )


def _resolve_fake_interval_method(
    requested: str | None, *, model: str, status: Status, censoring: Censoring, clustered: bool
) -> str:
    """The interval method of an interval-bearing fake threshold: ``"cell_bootstrap"`` for an
    isotonic one, ``"cluster_bootstrap"`` for a clustered one, ``"profile"`` otherwise, or the
    method asked for. A method is accepted only where the fit takes it and the result carries an
    interval."""
    default = (
        "cluster_bootstrap" if clustered else "cell_bootstrap" if model == "isotonic" else "profile"
    )
    if requested is None:
        return default
    takes = (
        _CLUSTERED_INTERVAL_METHODS
        if clustered
        else {
            "binomial": _MODEL_INTERVAL_METHODS | _INDEPENDENT_BOOTSTRAP_METHODS,
            "gaussian": _MODEL_INTERVAL_METHODS | {"parametric_bootstrap"},
            "isotonic": frozenset({"cell_bootstrap"}),
        }[model]
    )
    if requested not in takes:
        raise ValueError(
            f"fake_threshold: interval_method must be None or one of {sorted(takes)} for a "
            f"{model} threshold, got {requested!r}"
        )
    if status is not Status.OK or censoring not in (
        Censoring.NONE,
        Censoring.OPEN_UPPER,
        Censoring.OPEN_LOWER,
    ):
        raise ValueError(
            f"fake_threshold: interval_method={requested!r} needs an OK threshold with an "
            f"interval (censoring NONE, OPEN_UPPER or OPEN_LOWER), got status={status!r}, "
            f"censoring={censoring!r}"
        )
    return requested


def _fake_cluster_warnings(fit: Fit, interval_method: str) -> tuple[str, ...]:
    """The ``"FEW_CLUSTERS: "`` note a fake threshold carries, worded as the real threshold words
    it: the percentile cluster-bootstrap text for a ``"cluster_bootstrap"`` interval, the sandwich
    text otherwise. Empty for a fit without clusters."""
    if fit.cluster_ids is None:
        return ()
    if interval_method == "cluster_bootstrap":
        return (_few_clusters_warning(len(fit.cluster_ids), bootstrap=True),)
    return tuple(w for w in fit.warnings if w.startswith("FEW_CLUSTERS: "))


def _fake_ratio_bootstrap(lo: float | None, hi: float | None) -> BootstrapSummary:
    """The summary a fake paired ratio carries: clean, with the replicates counted on each side
    its interval leaves open (a lower end that is absent or zero, an upper end that is absent)."""
    return BootstrapSummary(
        n_resamples=999,
        failed=0,
        unreachable_low=40 if lo is None or lo == 0.0 else 0,
        unreachable_high=40 if hi is None else 0,
        seed=0,
    )


def _fake_bootstrap(method: str, censoring: Censoring) -> BootstrapSummary | None:
    """The summary a fake bootstrap threshold carries: clean, but with the side its open interval
    needs counted when it is open. ``None`` for any other method."""
    if method not in _BOOTSTRAP_INTERVAL_METHODS:
        return None
    return BootstrapSummary(
        n_resamples=999,
        failed=0,
        unreachable_low=40 if censoring is Censoring.OPEN_LOWER else 0,
        unreachable_high=40 if censoring is Censoring.OPEN_UPPER else 0,
        seed=0,
    )


def _gaussian_fake_threshold(
    *,
    fit: Fit,
    status: Status,
    censoring: Censoring,
    provenance: Mapping[str, JSONValue] | None,
    interval_method: str,
) -> Threshold:
    value: float | None = None
    lo: float | None = None
    hi: float | None = None
    method = "exact_bound"
    if status is Status.OK:
        value, method = 1.0, interval_method
        lo = None if censoring is Censoring.OPEN_LOWER else 0.8
        hi = None if censoring is Censoring.OPEN_UPPER else 1.25
    return Threshold(
        axis=_AXIS,
        definition=Definition.absolute(6.0),
        direction=fit.direction,
        value=value,
        lo=lo,
        hi=hi,
        level=0.95,
        interval_method=method,
        censoring=censoring,
        status=status,
        dependence="independent",
        fit=fit,
        warnings=(_FAKE_WARNING,),
        provenance=_fake_provenance(provenance),
        bootstrap=_fake_bootstrap(method, censoring),
    )


def _fake_ok_threshold(*, fit: Fit, value: float, lo: float, hi: float) -> Threshold:
    clustered = fit.dependence == "clustered"
    return Threshold(
        axis=_AXIS,
        definition=Definition.absolute(0.95),
        direction="decreasing",
        value=value,
        lo=lo,
        hi=hi,
        level=0.95,
        interval_method="cluster_bootstrap" if clustered else "profile",
        censoring=Censoring.NONE,
        status=Status.OK,
        dependence=fit.dependence,
        fit=fit,
        warnings=(
            _FAKE_WARNING,
            *_fake_cluster_warnings(fit, "cluster_bootstrap" if clustered else "profile"),
        ),
        provenance={},
        bootstrap=_fake_bootstrap("cluster_bootstrap", Censoring.NONE) if clustered else None,
    )


def fake_ratio(
    *,
    model: str = "binomial",
    shape: IntervalShape | None = None,
    status: Status = Status.OK,
    censoring: Censoring = Censoring.NONE,
    provenance: Mapping[str, JSONValue] | None = None,
    dependence: str = "independent",
) -> Ratio:
    """Build a deterministic, schema-valid :class:`~robustmargin.Ratio`.

    A failed or censored ratio carries no numbers: whenever ``status`` is not
    ``OK`` or ``censoring`` is not ``NONE``, ``shape``, ``estimate``, ``lo`` and ``hi`` are all
    ``None``, matching :class:`~robustmargin.Ratio`'s own invariant.

    Parameters
    ----------
    model
        ``"binomial"`` (the default) or ``"gaussian"``. A Gaussian ratio is of two Gaussian fake
        thresholds (see :func:`fake_threshold`); both are built on the same Gaussian fake cells,
        so unlike the binomial one their data are not disjoint.
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
    dependence
        ``"independent"`` (the default) or ``"paired"``. A paired fake is a ``"cluster_bootstrap"``
        ratio of two clustered binomial thresholds from two fits on one set of cluster ids
        (see :func:`fake_fit`), carrying a :class:`~robustmargin.BootstrapSummary` when it has an
        interval. ``EXCLUSIVE`` is a Fieller shape, not a bootstrap one, so it is not buildable
        here.

    Returns
    -------
    Ratio
        A ratio built from two fits on disjoint data (different cells entirely), with the given
        ``dependence``, ``cluster_ids=None`` on both fits unless it is ``"paired"``, and
        ``warnings`` naming it a fake. On the ``OK`` + ``NONE`` path, ``estimate`` is exactly
        ``threshold_a.value / threshold_b.value``, matching
        :class:`~robustmargin.Ratio`'s own invariant; the point estimate always lies inside the
        Fieller confidence set described by ``shape``, including for ``EXCLUSIVE``.

    Raises
    ------
    ValueError
        If ``model`` is not ``"binomial"`` or ``"gaussian"``, ``dependence`` is not
        ``"independent"`` or ``"paired"`` (or is ``"paired"`` with ``model="gaussian"``), or
        ``shape`` is given explicitly while ``status`` is not ``OK`` or ``censoring`` is not
        ``NONE``, or is ``EXCLUSIVE`` for a paired fake.
    """
    _check_model(model, function="fake_ratio", allowed=frozenset({"binomial", "gaussian"}))
    _check_dependence(dependence, function="fake_ratio", valid=_VALID_RATIO_DEPENDENCE)
    paired = dependence == "paired"
    if paired and model != "binomial":
        raise ValueError(
            f"fake_ratio: dependence='paired' needs model='binomial', got {model!r}: clustered "
            "continuous inference is not implemented"
        )
    if paired and shape is IntervalShape.EXCLUSIVE:
        raise ValueError(
            "fake_ratio: IntervalShape.EXCLUSIVE is a Fieller shape and a paired fake is a "
            "cluster-bootstrap ratio, whose interval is never two disjoint rays"
        )
    is_ok_none = status is Status.OK and censoring is Censoring.NONE
    if not is_ok_none and shape is not None:
        raise ValueError(
            f"fake_ratio: shape={shape!r} is not allowed when status={status!r} or "
            f"censoring={censoring!r} -- a failed or censored ratio carries no shape"
        )

    value_a, lo_a, hi_a = 0.05, 0.03, 0.08
    value_b, lo_b, hi_b = 0.1, 0.06, 0.16
    if model == "gaussian":
        base = fake_threshold(model="gaussian")
        threshold_a = dataclasses.replace(base, value=value_a, lo=lo_a, hi=hi_a)
        threshold_b = dataclasses.replace(base, value=value_b, lo=lo_b, hi=hi_b)
    else:
        threshold_a = _fake_ok_threshold(
            fit=_fake_ok_fit(_CELLS_A, dependence="clustered" if paired else "independent"),
            value=value_a,
            lo=lo_a,
            hi=hi_a,
        )
        threshold_b = _fake_ok_threshold(
            fit=_fake_ok_fit(_CELLS_B, dependence="clustered" if paired else "independent"),
            value=value_b,
            lo=lo_b,
            hi=hi_b,
        )

    cluster_warnings = (
        _fake_cluster_warnings(threshold_a.fit, "cluster_bootstrap") if paired else ()
    )
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
            # EXCLUSIVE: an analytic Fieller shape that needs a nonzero covariance term, so
            # ratio_interval() never gives it for an independent ratio and a paired fake is
            # a bootstrap one that cannot carry it. This fake builds it directly so a consumer
            # can test against it. The point estimate sits on the near edge of the excluded
            # gap, inside the (-inf, lo] ray of the Fieller confidence set.
            lo, hi = ratio_value, ratio_value + 1.0

    return Ratio(
        threshold_a=threshold_a,
        threshold_b=threshold_b,
        estimate=estimate,
        lo=lo,
        hi=hi,
        shape=resolved_shape,
        method="cluster_bootstrap" if paired else "fieller",
        dependence=dependence,
        level=0.95,
        censoring=censoring,
        status=status,
        warnings=(_FAKE_WARNING, *cluster_warnings),
        provenance=_fake_provenance(provenance),
        bootstrap=_fake_ratio_bootstrap(lo, hi) if paired and is_ok_none else None,
    )
