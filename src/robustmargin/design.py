"""Planning a sweep before running it: severity grids and the width of the interval to expect.

:func:`severity_grid` builds a severity grid on an axis. :func:`plan_threshold_width` and
:func:`plan_ratio_width` answer "if the system behaves like this pilot fit, how wide will the
confidence interval be at this design?". Each takes the point estimates of a converged binomial
:class:`~robustmargin.Fit` (for example one made on a coarse pilot sweep) as the truth, simulates
whole experiments at a candidate design, refits the pilot's model on each and calls the real
:func:`~robustmargin.threshold` (and :func:`~robustmargin.ratio_interval`) with the caller's
methods. The planner therefore measures the analysis that will be reported, not an approximation
of it. The result is a :class:`WidthPlan`, one width per simulated experiment.

**Data-generating process.** Without clusters, the successes at each level are independent
binomial draws with the pilot curve's success probability ``P(x)``. With ``n_clusters = G``
each cluster is observed at every level with ``trials_per_level / G`` trials, and a trial in
cluster ``g`` is a success when ``Phi(Y) < P(x)``, where ``Phi`` is the standard normal CDF and
``Y = sqrt(icc) * b_g + sqrt(1 - icc) * e`` with ``b_g`` and ``e`` independent standard normal
(a Gaussian copula with a latent threshold; see Nelsen, 2006, *An Introduction to Copulas*,
2nd ed., Springer). Each level's marginal success probability is exactly the pilot curve whatever
``icc`` is, so the quantity being estimated does not move with the correlation; ``icc`` is the
latent correlation between two trials of one cluster, which maps to a smaller correlation of the
binary outcomes. The cluster effect ``b_g`` is shared by all levels of a cluster. In a paired
ratio the two arms share the same clusters and the same ``b_g``; in an independent ratio they
have disjoint clusters.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr, ndtri

from robustmargin.models import Fit, _curve_success_probability, fit_dose_response
from robustmargin.ratio import _VALID_DEPENDENCE, _VALID_METHODS, ratio_interval
from robustmargin.threshold import (
    _BOOTSTRAP_METHODS,
    _CLUSTERED_INTERVAL_METHODS,
    _VALID_INTERVAL_METHOD_REQUESTS,
    Threshold,
    threshold,
)
from robustmargin.types import Axis, Censoring, Definition, IntervalShape, Observations, Status

__all__ = ["WidthPlan", "plan_ratio_width", "plan_threshold_width", "severity_grid"]

_PLAN_SCHEMA_VERSION = "1"
_KINDS = frozenset({"threshold", "ratio"})
_WIDTH_SCALES = frozenset({"log_ratio", "difference"})
_THRESHOLD_DEPENDENCE = frozenset({"independent", "clustered"})
_RATIO_DEPENDENCE = _VALID_DEPENDENCE
_OUTCOME_OK = "OK"
_OUTCOME_BOTH_SIDES_OPEN = "BOTH_SIDES_OPEN"
_OUTCOME_LOWER_AT_ZERO = "LOWER_AT_ZERO"
_OUTCOME_LABELS = frozenset(
    {_OUTCOME_OK, _OUTCOME_BOTH_SIDES_OPEN, _OUTCOME_LOWER_AT_ZERO}
    | {status.value for status in Status if status is not Status.OK}
    | {censoring.value for censoring in Censoring if censoring is not Censoring.NONE}
    | {shape.value for shape in IntervalShape if shape is not IntervalShape.BOUNDED}
)


# --------------------------------------------------------------------------------------------
# small validators
# --------------------------------------------------------------------------------------------


def _check_int(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}: {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}")
    return int(value)


def _check_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


# --------------------------------------------------------------------------------------------
# severity_grid
# --------------------------------------------------------------------------------------------


def severity_grid(
    axis: Axis,
    *,
    lowest: float,
    highest: float,
    n_levels: int,
    control: bool = True,
) -> tuple[float, ...]:
    """A strictly increasing severity grid on ``axis``, optionally with a zero-severity control.

    A log axis gets ``n_levels`` geometrically spaced levels from ``lowest`` to ``highest``, and a
    linear axis arithmetically spaced ones. With ``control=True`` a leading ``0.0`` is prepended,
    except on a linear axis whose ``lowest`` is already ``0.0`` (a second zero would repeat the
    level). The end points are exactly ``lowest`` and ``highest``, not the round-off of the
    spacing function.

    Parameters
    ----------
    axis
        The severity axis; its ``scale`` picks geometric or arithmetic spacing.
    lowest, highest
        The first and last non-control level. ``highest`` must exceed ``lowest``; on a log axis
        ``lowest`` must be positive, and on a linear axis non-negative.
    n_levels
        The number of levels from ``lowest`` to ``highest`` inclusive, at least 2. The control, if
        added, is one more.
    control
        Whether to prepend the zero-severity control.

    Returns
    -------
    tuple[float, ...]
        The grid, as plain Python floats.

    Raises
    ------
    TypeError
        If ``n_levels`` is not an int or ``lowest``/``highest`` is not a number.
    ValueError
        If ``n_levels < 2``, ``highest <= lowest``, ``lowest <= 0`` on a log axis, ``lowest < 0``
        on a linear axis, or either end is not finite.
    """
    n = _check_int(n_levels, name="severity_grid n_levels", minimum=2)
    low = _check_float(lowest, name="severity_grid lowest")
    high = _check_float(highest, name="severity_grid highest")
    if high <= low:
        raise ValueError(f"severity_grid: highest ({high}) must exceed lowest ({low})")
    if axis.scale == "log":
        if low <= 0.0:
            raise ValueError(f"severity_grid: lowest must be positive on a log axis, got {low}")
        levels = np.geomspace(low, high, n)
    else:
        if low < 0.0:
            raise ValueError(f"severity_grid: lowest must be non-negative, got {low}")
        levels = np.linspace(low, high, n)
    levels[0] = low
    levels[-1] = high
    grid = [float(v) for v in levels]
    if control and not (axis.scale != "log" and low == 0.0):
        grid.insert(0, 0.0)
    return tuple(grid)


# --------------------------------------------------------------------------------------------
# WidthPlan
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class WidthPlan:
    """The widths of the confidence interval a design would give, one per simulated experiment.

    Built by :func:`plan_threshold_width` and :func:`plan_ratio_width`; it records the design, the
    methods and every replicate's outcome. A replicate that gives no finite width is never
    dropped: ``widths[i]`` is ``None`` and ``outcomes[i]`` says why, so every summary treats it
    as an infinite width.

    Attributes
    ----------
    kind
        ``"threshold"`` or ``"ratio"``.
    axis
        The :class:`~robustmargin.Axis` of the pilot fit.
    severity
        The severity levels of the design (a control, if any, included).
    trials_per_level
        Trials at each level, summed over clusters.
    n_clusters
        Clusters per arm, or ``None`` for independent trials.
    icc
        The latent within-cluster correlation of the data-generating process (``0.0`` without
        clusters). It is the correlation of the underlying normal variables, not of the binary
        outcomes; see :func:`plan_threshold_width`.
    definition
        The :class:`~robustmargin.Definition` of the threshold being planned.
    interval_method
        The threshold interval method the replicates used.
    ratio_method
        The ratio interval method, or ``None`` for a threshold result.
    dependence
        ``"independent"`` or ``"clustered"`` for a threshold result; ``"independent"`` or
        ``"paired"`` for a ratio result.
    level
        The confidence level of the intervals.
    n_replicates
        The number of simulated experiments; equals ``len(widths)``.
    seed
        The seed the replicate streams were spawned from.
    n_resamples
        The number of bootstrap resamples per interval, or ``None`` for the library default (and
        for a method that does not bootstrap).
    width_scale
        ``"log_ratio"`` (width is ``log(hi / lo)``: every ratio and every log-axis threshold) or
        ``"difference"`` (width is ``hi - lo``: a linear-axis threshold).
    widths
        One entry per replicate: the finite, non-negative width, or ``None`` if the replicate
        gave no bounded interval.
    outcomes
        One label per replicate. ``"OK"`` exactly when the width is finite. Otherwise the first
        that applies of: the :class:`~robustmargin.Status` value when the fit, threshold or ratio
        is not ``OK`` (for example ``"NOT_CONVERGED"``); the :class:`~robustmargin.Censoring` value
        when the result is censored (``"OPEN_UPPER"``, ``"RIGHT"``, ...); for a ratio, the
        :class:`~robustmargin.IntervalShape` value when it is not ``BOUNDED``;
        ``"BOTH_SIDES_OPEN"`` for a threshold that is not censored but has no finite end on
        either side; ``"LOWER_AT_ZERO"`` for a bounded ratio whose lower end is not positive,
        which has no finite log-ratio width. These are the only labels.
    plan_schema_version
        The serialised version of a :class:`WidthPlan`, ``"1"``.
    library_version
        The robustmargin version that produced the result.
    provenance
        An opaque mapping stored as given and never interpreted. Empty by default.

    Raises
    ------
    TypeError
        On construction, if a field has the wrong type, including a ``bool`` where a number is
        expected.
    ValueError
        On construction, if a field is outside what the planners produce: ``widths`` and
        ``outcomes`` do not each have ``n_replicates`` entries, ``outcomes[i] == "OK"`` is not
        equivalent to ``widths[i] is not None``, an outcome is not one of the labels above, a
        finite width is negative or not finite, ``width_scale`` does not match the kind and axis
        (a ratio and a log-axis threshold are ``"log_ratio"``, a linear-axis threshold
        ``"difference"``), or a design field is out of range.
    """

    kind: str
    axis: Axis
    severity: tuple[float, ...]
    trials_per_level: int
    n_clusters: int | None
    icc: float
    definition: Definition
    interval_method: str
    ratio_method: str | None
    dependence: str
    level: float
    n_replicates: int
    seed: int
    n_resamples: int | None
    width_scale: str
    widths: tuple[float | None, ...]
    outcomes: tuple[str, ...]
    plan_schema_version: str = _PLAN_SCHEMA_VERSION
    library_version: str = field(default_factory=lambda: _library_version())
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"WidthPlan.kind must be one of {sorted(_KINDS)}, got {self.kind!r}")
        if not isinstance(self.axis, Axis):
            raise TypeError(f"WidthPlan.axis must be an Axis, got {type(self.axis).__name__}")
        if not isinstance(self.definition, Definition):
            raise TypeError(
                f"WidthPlan.definition must be a Definition, got {type(self.definition).__name__}"
            )
        if isinstance(self.severity, (str, bytes)) or not isinstance(self.severity, Sequence):
            raise TypeError("WidthPlan.severity must be a sequence of numbers")
        severity = tuple(_check_float(v, name="WidthPlan.severity entry") for v in self.severity)
        if any(v < 0.0 for v in severity):
            raise ValueError("WidthPlan.severity must be non-negative")
        if any(b <= a for a, b in zip(severity, severity[1:], strict=False)):
            raise ValueError("WidthPlan.severity must be strictly increasing")
        if sum(1 for v in severity if v > 0.0) < 2:
            raise ValueError("WidthPlan.severity needs at least two nonzero levels")
        object.__setattr__(self, "severity", severity)
        _check_int(self.trials_per_level, name="WidthPlan.trials_per_level", minimum=1)
        if self.n_clusters is not None:
            _check_int(self.n_clusters, name="WidthPlan.n_clusters", minimum=2)
            if self.trials_per_level % self.n_clusters != 0:
                raise ValueError("WidthPlan.trials_per_level must be divisible by n_clusters")
        icc = _check_float(self.icc, name="WidthPlan.icc")
        if not 0.0 <= icc < 1.0:
            raise ValueError(f"WidthPlan.icc must be in [0, 1), got {icc!r}")
        object.__setattr__(self, "icc", icc)
        if icc > 0.0 and self.n_clusters is None:
            raise ValueError("WidthPlan.icc > 0 needs n_clusters")
        if self.interval_method not in _VALID_INTERVAL_METHOD_REQUESTS:
            raise ValueError(
                f"WidthPlan.interval_method must be one of "
                f"{sorted(_VALID_INTERVAL_METHOD_REQUESTS)}, got {self.interval_method!r}"
            )
        if self.kind == "threshold":
            if self.ratio_method is not None:
                raise ValueError("WidthPlan.ratio_method must be None for kind 'threshold'")
            allowed = _THRESHOLD_DEPENDENCE
        else:
            if self.ratio_method not in _VALID_METHODS:
                raise ValueError(
                    f"WidthPlan.ratio_method must be one of {sorted(_VALID_METHODS)} for kind "
                    f"'ratio', got {self.ratio_method!r}"
                )
            allowed = _RATIO_DEPENDENCE
        if self.dependence not in allowed:
            raise ValueError(
                f"WidthPlan.dependence for kind {self.kind!r} must be one of {sorted(allowed)}, "
                f"got {self.dependence!r}"
            )
        if self.dependence in ("clustered", "paired") and self.n_clusters is None:
            raise ValueError(f"WidthPlan.dependence {self.dependence!r} needs n_clusters")
        if self.kind == "threshold" and (self.dependence == "clustered") != (
            self.n_clusters is not None
        ):
            raise ValueError("WidthPlan: a threshold is clustered exactly when it has n_clusters")
        level = _check_float(self.level, name="WidthPlan.level")
        if not 0.0 < level < 1.0:
            raise ValueError(f"WidthPlan.level must be strictly between 0 and 1, got {level!r}")
        object.__setattr__(self, "level", level)
        _check_int(self.n_replicates, name="WidthPlan.n_replicates", minimum=1)
        _check_int(self.seed, name="WidthPlan.seed", minimum=0)
        if self.n_resamples is not None:
            _check_int(self.n_resamples, name="WidthPlan.n_resamples", minimum=1)
        if self.width_scale not in _WIDTH_SCALES:
            raise ValueError(
                f"WidthPlan.width_scale must be one of {sorted(_WIDTH_SCALES)}, "
                f"got {self.width_scale!r}"
            )
        expected_scale = (
            "difference"
            if self.kind == "threshold" and self.axis.scale == "linear"
            else "log_ratio"
        )
        if self.width_scale != expected_scale:
            raise ValueError(
                f"WidthPlan.width_scale must be {expected_scale!r} for kind {self.kind!r} on a "
                f"{self.axis.scale} axis, got {self.width_scale!r}"
            )
        if self.plan_schema_version != _PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"WidthPlan.plan_schema_version must be {_PLAN_SCHEMA_VERSION!r}, "
                f"got {self.plan_schema_version!r}"
            )
        if not isinstance(self.library_version, str):
            raise TypeError(
                "WidthPlan.library_version must be a str, "
                f"got {type(self.library_version).__name__}"
            )
        widths = tuple(self.widths)
        outcomes = tuple(self.outcomes)
        object.__setattr__(self, "widths", widths)
        object.__setattr__(self, "outcomes", outcomes)
        if not (len(widths) == len(outcomes) == self.n_replicates):
            raise ValueError(
                "WidthPlan: widths, outcomes and n_replicates must agree, got "
                f"{len(widths)}, {len(outcomes)} and {self.n_replicates}"
            )
        for index, (width, outcome) in enumerate(zip(widths, outcomes, strict=True)):
            if not isinstance(outcome, str) or outcome not in _OUTCOME_LABELS:
                raise ValueError(
                    f"WidthPlan.outcomes[{index}] must be one of {sorted(_OUTCOME_LABELS)}, "
                    f"got {outcome!r}"
                )
            if (outcome == _OUTCOME_OK) != (width is not None):
                raise ValueError(
                    f"WidthPlan: replicate {index} has outcome {outcome!r} and width {width!r}; "
                    "'OK' goes with a finite width and every other outcome with None"
                )
            if width is not None:
                value = _check_float(width, name=f"WidthPlan.widths[{index}]")
                if value < 0.0:
                    raise ValueError(
                        f"WidthPlan.widths[{index}] must be non-negative, got {width!r}"
                    )

    def outcome_counts(self) -> dict[str, int]:
        """The number of replicates with each outcome label, ``"OK"`` included.

        Returns
        -------
        dict[str, int]
            Label to count; the counts sum to ``n_replicates``.
        """
        counts: dict[str, int] = {}
        for outcome in self.outcomes:
            counts[outcome] = counts.get(outcome, 0) + 1
        return counts

    def quantile(self, q: float) -> float:
        """The ``q`` quantile of the widths, with a replicate that has no finite width counted as
        an infinite one.

        Linear interpolation between order statistics (the default of ``numpy.quantile``). The
        result is ``math.inf`` when the quantile falls among the replicates with no finite width.

        Parameters
        ----------
        q
            A probability in ``[0, 1]``.

        Returns
        -------
        float
            The quantile, in the units of ``width_scale``.

        Raises
        ------
        ValueError
            If ``q`` is not in ``[0, 1]``.
        """
        if not (isinstance(q, (int, float)) and not isinstance(q, bool) and 0.0 <= q <= 1.0):
            raise ValueError(f"WidthPlan.quantile: q must be in [0, 1], got {q!r}")
        ordered = sorted(math.inf if w is None else w for w in self.widths)
        position = q * (len(ordered) - 1)
        lower = math.floor(position)
        fraction = position - lower
        if fraction == 0.0 or lower + 1 >= len(ordered):
            return ordered[lower]
        upper_value = ordered[lower + 1]
        if math.isinf(upper_value):
            return math.inf
        return ordered[lower] * (1.0 - fraction) + upper_value * fraction

    def share_at_most(self, width: float) -> tuple[float, float]:
        """The share of replicates whose width is finite and at most ``width``, with its Monte
        Carlo standard error.

        This answers "how often does this design give an interval no wider than ``width``?". A
        replicate with no finite width never counts, however large ``width`` is.

        Parameters
        ----------
        width
            The widest acceptable width, in the units of ``width_scale``.

        Returns
        -------
        tuple[float, float]
            The share ``p`` and ``sqrt(p * (1 - p) / n_replicates)``, the binomial plug-in standard
            error of ``p`` over the simulated replicates.

        Notes
        -----
        The standard error is 0 when ``p`` is 0 or 1, which understates the uncertainty there: a
        share of 1 from 20 replicates does not show that the true share is 1. Run more replicates
        before relying on a share at either extreme.
        """
        n = len(self.widths)
        hits = sum(1 for w in self.widths if w is not None and w <= width)
        share = hits / n
        return share, math.sqrt(share * (1.0 - share) / n)


# --------------------------------------------------------------------------------------------
# the data-generating process
# --------------------------------------------------------------------------------------------


def _success_probability(curve: Fit, severity: NDArray[np.float64]) -> NDArray[np.float64]:
    """The pilot curve's probability of the counted outcome at each severity.

    An increasing curve is the complement of the decreasing curve with asymptotes ``1 - lower`` and
    ``1 - upper``, so the same evaluator is used on that complement.
    """
    assert curve.params is not None
    mu, s = curve.params["mu"].value, curve.params["s"].value
    upper, lower = curve.params["upper"].value, curve.params["lower"].value
    if curve.direction == "increasing":
        complement = _curve_success_probability(
            severity,
            mu=mu,
            s=s,
            upper=1.0 - lower,
            lower=1.0 - upper,
            scale=curve.axis.scale,
            link=curve.link,
        )
        p = 1.0 - complement
        if curve.axis.scale == "log":
            p = np.where(severity == 0.0, lower, p)
    else:
        p = _curve_success_probability(
            severity,
            mu=mu,
            s=s,
            upper=upper,
            lower=lower,
            scale=curve.axis.scale,
            link=curve.link,
        )
    return np.clip(p, 0.0, 1.0)


def _simulate_counts(
    curves: Sequence[Fit],
    severity: Sequence[float],
    trials_per_level: int,
    *,
    rng: np.random.Generator,
    n_clusters: int | None = None,
    icc: float = 0.0,
    shared_effects: bool = False,
) -> NDArray[np.int64]:
    """Successes drawn from the pilot curves at a design, shape ``(n_arms, n_levels, G)``.

    ``G`` is ``n_clusters`` (1 without clusters) and each cell has ``trials_per_level // G``
    trials. A trial in a cluster with effect ``b`` succeeds when ``Phi(sqrt(icc) * b +
    sqrt(1 - icc) * e) < P(x)`` for a fresh standard normal ``e``; given ``b`` the successes of a
    cell are therefore binomial with probability ``Phi((Phi^-1(P) - sqrt(icc) * b) /
    sqrt(1 - icc))``, which is what is drawn. With ``shared_effects`` the arms use the same ``b``.
    """
    clusters = 1 if n_clusters is None else n_clusters
    per_cell = trials_per_level // clusters
    x = np.asarray(severity, dtype=np.float64)
    p = np.stack([_success_probability(curve, x) for curve in curves])  # (arms, levels)
    arms = p.shape[0]
    if icc == 0.0:
        cell_p = np.broadcast_to(p[:, :, None], (arms, x.size, clusters))
    else:
        shape = (1, clusters) if shared_effects else (arms, clusters)
        effects = rng.standard_normal(shape)
        shift = math.sqrt(icc) * effects[:, None, :]  # (1 or arms, 1, G)
        with np.errstate(invalid="ignore"):
            cell_p = ndtr((ndtri(p)[:, :, None] - shift) / math.sqrt(1.0 - icc))
    return rng.binomial(per_cell, cell_p).astype(np.int64)


# --------------------------------------------------------------------------------------------
# one replicate
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Spec:
    """Everything a replicate needs; plain data so it pickles to a worker process."""

    curves: tuple[Fit, ...]
    severity: tuple[float, ...]
    trials_per_level: int
    definition: Definition
    interval_method: str
    ratio_method: str | None
    dependence: str
    level: float
    n_clusters: int | None
    icc: float
    n_resamples: int | None


def _observations(
    curve: Fit,
    severity: Sequence[float],
    counts: NDArray[np.int64],
    *,
    trials: int,
    cluster_offset: int,
    clustered: bool,
) -> Observations:
    levels, clusters = counts.shape
    rows_severity: list[float] = []
    rows_successes: list[int] = []
    rows_cluster: list[int] = []
    for j in range(levels):
        for g in range(clusters):
            rows_severity.append(float(severity[j]))
            rows_successes.append(int(counts[j, g]))
            rows_cluster.append(cluster_offset + g)
    return Observations.from_counts(
        curve.axis,
        severity=rows_severity,
        successes=rows_successes,
        trials=[trials] * len(rows_severity),
        outcome=curve.outcome,
        direction=curve.direction,
        cluster=rows_cluster if clustered else None,
    )


def _fit_arm(curve: Fit, observations: Observations, *, clustered: bool) -> Fit:
    assert curve.params is not None
    upper_param, lower_param = curve.params["upper"], curve.params["lower"]
    upper: float | Literal["estimate"] = upper_param.value if upper_param.fixed else "estimate"
    lower: float | Literal["estimate"] = lower_param.value if lower_param.fixed else "estimate"
    link: Literal["probit", "logit", "cloglog"]
    if curve.link == "probit":
        link = "probit"
    elif curve.link == "logit":
        link = "logit"
    elif curve.link == "cloglog":
        link = "cloglog"
    else:
        raise ValueError(f"cannot refit a binomial fit with link {curve.link!r}")
    return fit_dose_response(
        observations,
        model="binomial",
        link=link,
        upper=upper,
        lower=lower,
        dependence="clustered" if clustered else None,
    )


def _threshold_width(t: Threshold, *, log_scale: bool) -> tuple[float | None, str]:
    if t.status is not Status.OK:
        return None, t.status.value
    if t.censoring is not Censoring.NONE:
        return None, t.censoring.value
    if t.lo is None or t.hi is None:
        return None, _OUTCOME_BOTH_SIDES_OPEN
    if log_scale:
        return math.log(t.hi / t.lo), _OUTCOME_OK
    return t.hi - t.lo, _OUTCOME_OK


def _bootstrap_seed(rng: np.random.Generator, method: str | None) -> int | None:
    """A seed for one bootstrap call from the replicate's own stream, or ``None``."""
    if method in _BOOTSTRAP_METHODS:
        return int(rng.integers(2**63 - 1))
    return None


def _run_replicate(spec: _Spec, stream: np.random.SeedSequence) -> tuple[float | None, str]:
    """One simulated experiment: draw data, refit, threshold (and ratio), and score the width."""
    rng = np.random.default_rng(stream)
    paired = spec.dependence == "paired"
    clusters = spec.n_clusters
    counts = _simulate_counts(
        spec.curves,
        spec.severity,
        spec.trials_per_level,
        rng=rng,
        n_clusters=clusters,
        icc=spec.icc,
        shared_effects=paired,
    )
    per_cell = spec.trials_per_level // (1 if clusters is None else clusters)
    clustered = clusters is not None
    thresholds: list[Threshold] = []
    for arm, curve in enumerate(spec.curves):
        offset = 0 if paired or clusters is None else arm * clusters
        observations = _observations(
            curve,
            spec.severity,
            counts[arm],
            trials=per_cell,
            cluster_offset=offset,
            clustered=clustered,
        )
        fit = _fit_arm(curve, observations, clustered=clustered)
        if fit.status is not Status.OK:
            return None, fit.status.value
        thresholds.append(
            threshold(
                fit,
                definition=spec.definition,
                interval_method=spec.interval_method,
                level=spec.level,
                dependence="clustered" if clustered else "independent",
                n_resamples=spec.n_resamples,
                seed=_bootstrap_seed(rng, spec.interval_method),
            )
        )
    if spec.ratio_method is None:
        return _threshold_width(thresholds[0], log_scale=spec.curves[0].axis.scale == "log")
    ratio = ratio_interval(
        thresholds[0],
        thresholds[1],
        method=spec.ratio_method,
        dependence=spec.dependence,
        level=spec.level,
        n_resamples=spec.n_resamples,
        seed=_bootstrap_seed(rng, spec.ratio_method),
    )
    if ratio.status is not Status.OK:
        return None, ratio.status.value
    if ratio.censoring is not Censoring.NONE:
        return None, ratio.censoring.value
    if ratio.shape is not None and ratio.shape is not IntervalShape.BOUNDED:
        return None, ratio.shape.value
    if ratio.lo is None or ratio.hi is None:
        return None, _OUTCOME_BOTH_SIDES_OPEN
    if ratio.lo <= 0.0:
        return None, _OUTCOME_LOWER_AT_ZERO
    return math.log(ratio.hi / ratio.lo), _OUTCOME_OK


def _run_replicate_task(task: tuple[_Spec, np.random.SeedSequence]) -> tuple[float | None, str]:
    return _run_replicate(*task)


# --------------------------------------------------------------------------------------------
# public planners
# --------------------------------------------------------------------------------------------


def _check_pilot(curve: Fit, *, name: str) -> None:
    if not isinstance(curve, Fit):
        raise TypeError(f"{name} must be a Fit, got {type(curve).__name__}")
    if curve.model != "binomial":
        raise ValueError(
            f"{name}: only binomial fits can be planned from, got model {curve.model!r}"
        )
    if curve.status is not Status.OK:
        raise ValueError(f"{name} must have status OK, got {curve.status.value}")
    if curve.params is None:
        raise ValueError(f"{name} has no fitted parameters to simulate from")


def _check_design(
    severity: Sequence[float],
    trials_per_level: int,
    n_clusters: int | None,
    icc: float,
    level: float,
    n_replicates: int,
    seed: int,
    workers: int,
    definition: Definition,
    n_resamples: int | None,
    interval_method: str,
    ratio_method: str | None,
) -> tuple[tuple[float, ...], float, float]:
    if interval_method not in _VALID_INTERVAL_METHOD_REQUESTS:
        raise ValueError(
            f"interval_method must be one of {sorted(_VALID_INTERVAL_METHOD_REQUESTS)}, "
            f"got {interval_method!r}"
        )
    if n_clusters is not None and interval_method not in _CLUSTERED_INTERVAL_METHODS:
        raise ValueError(
            f"a clustered design needs interval_method in "
            f"{sorted(_CLUSTERED_INTERVAL_METHODS & _VALID_INTERVAL_METHOD_REQUESTS)}, "
            f"got {interval_method!r}"
        )
    if ratio_method is not None and ratio_method not in _VALID_METHODS:
        raise ValueError(
            f"ratio_method must be one of {sorted(_VALID_METHODS)}, got {ratio_method!r}"
        )
    if n_resamples is not None:
        _check_int(n_resamples, name="n_resamples", minimum=1)
        if interval_method not in _BOOTSTRAP_METHODS and ratio_method != "cluster_bootstrap":
            raise ValueError(
                "n_resamples applies only to a bootstrap interval_method or ratio_method, "
                f"got interval_method={interval_method!r}, ratio_method={ratio_method!r}"
            )
    _check_int(n_replicates, name="n_replicates", minimum=1)
    _check_int(workers, name="workers", minimum=1)
    _check_int(seed, name="seed", minimum=0)
    _check_int(trials_per_level, name="trials_per_level", minimum=1)
    if not isinstance(definition, Definition):
        raise TypeError(f"definition must be a Definition, got {type(definition).__name__}")
    levels = tuple(_check_float(v, name="severity entry") for v in severity)
    if any(v < 0.0 for v in levels):
        raise ValueError("severity must be non-negative")
    if any(b <= a for a, b in zip(levels, levels[1:], strict=False)):
        raise ValueError("severity must be strictly increasing")
    if sum(1 for v in levels if v > 0.0) < 2:
        raise ValueError("severity needs at least two distinct nonzero levels")
    if n_clusters is not None:
        _check_int(n_clusters, name="n_clusters", minimum=2)
        if trials_per_level % n_clusters != 0:
            raise ValueError(
                f"trials_per_level ({trials_per_level}) must be divisible by n_clusters "
                f"({n_clusters})"
            )
    icc_value = _check_float(icc, name="icc")
    if not 0.0 <= icc_value < 1.0:
        raise ValueError(f"icc must be in [0, 1), got {icc_value}")
    if icc_value > 0.0 and n_clusters is None:
        raise ValueError("icc > 0 needs n_clusters")
    level_value = _check_float(level, name="level")
    if not 0.0 < level_value < 1.0:
        raise ValueError(f"level must be strictly between 0 and 1, got {level_value}")
    return levels, icc_value, level_value


def _execute(
    spec: _Spec, *, n_replicates: int, seed: int, workers: int
) -> list[tuple[float | None, str]]:
    streams = np.random.SeedSequence(seed).spawn(n_replicates)
    tasks = [(spec, stream) for stream in streams]
    if workers == 1 or n_replicates == 1:
        return [_run_replicate_task(task) for task in tasks]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=min(workers, n_replicates)) as pool:
        return list(pool.map(_run_replicate_task, tasks))


def _library_version() -> str:
    import robustmargin

    return robustmargin.__version__


def _build_plan(
    spec: _Spec,
    results: list[tuple[float | None, str]],
    *,
    kind: str,
    axis: Axis,
    seed: int,
    provenance: Mapping[str, Any] | None,
) -> WidthPlan:
    linear_threshold = kind == "threshold" and axis.scale == "linear"
    return WidthPlan(
        kind=kind,
        axis=axis,
        severity=spec.severity,
        trials_per_level=spec.trials_per_level,
        n_clusters=spec.n_clusters,
        icc=spec.icc,
        definition=spec.definition,
        interval_method=spec.interval_method,
        ratio_method=spec.ratio_method,
        dependence=spec.dependence,
        level=spec.level,
        n_replicates=len(results),
        seed=seed,
        n_resamples=spec.n_resamples,
        width_scale="difference" if linear_threshold else "log_ratio",
        widths=tuple(width for width, _ in results),
        outcomes=tuple(label for _, label in results),
        provenance={} if provenance is None else provenance,
    )


def plan_threshold_width(
    curve: Fit,
    *,
    severity: Sequence[float],
    trials_per_level: int,
    definition: Definition,
    interval_method: str,
    level: float,
    n_replicates: int,
    seed: int,
    n_clusters: int | None = None,
    icc: float = 0.0,
    n_resamples: int | None = None,
    workers: int = 1,
    provenance: Mapping[str, Any] | None = None,
) -> WidthPlan:
    """The widths of a threshold's confidence interval to expect at a candidate design.

    Taking ``curve``'s point estimates as the truth, each of ``n_replicates`` experiments draws
    successes at ``severity`` (``trials_per_level`` trials at each level, split over
    ``n_clusters`` clusters when given; see the module description for the process), refits the
    pilot's specification (model, link, which asymptotes are fixed or estimated, and the fixed
    values) and calls :func:`~robustmargin.threshold` with ``definition``, ``interval_method`` and
    ``level``. The width is ``log(hi / lo)`` on a log axis and ``hi - lo`` on a linear one. A
    replicate whose fit or threshold is not ``OK``, is censored, or has no finite end on either
    side has no finite width and is tallied by reason in :attr:`WidthPlan.outcomes`; nothing is
    dropped. A clustered design is analysed with ``dependence="clustered"``.

    Replicate ``i`` uses the ``i``-th stream spawned from ``numpy.random.SeedSequence(seed)``,
    so a replicate does not depend on ``n_replicates`` or on ``workers``: the result is the same for
    any worker count. A bootstrap interval draws its own seed from the replicate's stream, so a
    bootstrap design is reproducible too. ``workers > 1`` runs replicates in separate processes
    (standard library ``concurrent.futures``); under the ``spawn`` start method (the default on
    Windows and macOS) the calling script must guard its entry point with
    ``if __name__ == "__main__":``. Each replicate is a real fit and interval; as a guide, about
    0.4 s with fixed asymptotes and a profile interval and about 2 s with estimated asymptotes on
    a laptop, and about 13 s with a bootstrap interval at the default 999 resamples with fixed
    asymptotes, so use ``workers`` for large runs.

    Parameters
    ----------
    curve
        A pilot :class:`~robustmargin.Fit`: binomial, ``Status.OK``. Its direction and outcome
        orientation are respected. An isotonic or Gaussian fit raises.
    severity
        The candidate design's levels, strictly increasing and non-negative, with at least two
        nonzero levels (a leading ``0.0`` is the control).
    trials_per_level
        Trials at each level, summed over clusters.
    definition, interval_method, level
        Passed to :func:`~robustmargin.threshold`.
    n_replicates
        The number of simulated experiments, at least 1.
    seed
        A non-negative integer.
    n_clusters
        Clusters, at least 2 and dividing ``trials_per_level``, or ``None`` for independent
        trials.
    icc
        The latent within-cluster correlation, in ``[0, 1)``, of the normal variables the
        copula is built from; it is not the intraclass correlation of the binary outcomes, which
        is smaller and depends on the success probability ``P`` of the level. With
        ``q = Phi^-1(P)``, the binary correlation is
        ``rho_bin(P) = [Phi2(q, q; icc) - P**2] / (P * (1 - P))``, where ``Phi2`` is the standard
        bivariate normal CDF with correlation ``icc``. For example a latent ``icc = 0.3`` gives
        a binary correlation of about 0.19 at ``P = 0.5`` and about 0.10 at ``P = 0.95``. In a
        paired ratio the latent correlation between the two arms equals ``icc`` (the arms share
        the cluster effect and have independent residuals) and cannot be set separately.
        Positive values need ``n_clusters``.
    n_resamples
        The number of bootstrap resamples per interval, passed to :func:`~robustmargin.threshold`
        (and, for a ratio, to :func:`~robustmargin.ratio_interval`). ``None`` takes the library
        default. It must be ``None`` unless a bootstrap method is used.
    workers
        The number of worker processes, at least 1.
    provenance
        An opaque mapping stored on the result as given.

    Returns
    -------
    WidthPlan
        The widths and outcomes of the replicates.

    Raises
    ------
    TypeError
        If an integer argument is a ``bool`` or not an integer.
    ValueError
        If ``curve`` is not a binomial ``Status.OK`` fit, ``severity`` is not strictly increasing
        and non-negative with two nonzero levels, ``trials_per_level`` is not divisible by
        ``n_clusters``, ``icc`` is outside ``[0, 1)`` or positive without clusters, ``level`` is
        not in ``(0, 1)``, ``interval_method`` is not a model or bootstrap interval method (or, with
        ``n_clusters``, not a clustered one), or ``n_resamples`` is given for a method that does
        not bootstrap.

    Notes
    -----
    The result is only as good as the pilot: the truth is the pilot's point estimates, so the width
    of an interval around a poorly estimated pilot curve is itself uncertain. A design with levels
    outside the pilot's tested range extrapolates the pilot's model there.
    """
    _check_pilot(curve, name="curve")
    levels, icc_value, level_value = _check_design(
        severity,
        trials_per_level,
        n_clusters,
        icc,
        level,
        n_replicates,
        seed,
        workers,
        definition,
        n_resamples,
        interval_method,
        None,
    )
    spec = _Spec(
        curves=(curve,),
        severity=levels,
        trials_per_level=trials_per_level,
        definition=definition,
        interval_method=interval_method,
        ratio_method=None,
        dependence="independent" if n_clusters is None else "clustered",
        level=level_value,
        n_clusters=n_clusters,
        icc=icc_value,
        n_resamples=n_resamples,
    )
    results = _execute(spec, n_replicates=n_replicates, seed=seed, workers=workers)
    return _build_plan(
        spec, results, kind="threshold", axis=curve.axis, seed=seed, provenance=provenance
    )


def _same_spec(curve_a: Fit, curve_b: Fit) -> None:
    if (curve_a.axis.name, curve_a.axis.unit, curve_a.axis.scale) != (
        curve_b.axis.name,
        curve_b.axis.unit,
        curve_b.axis.scale,
    ):
        raise ValueError(
            "plan_ratio_width: curve_a and curve_b must share axis name, unit and scale"
        )
    if curve_a.direction != curve_b.direction:
        raise ValueError("plan_ratio_width: curve_a and curve_b must share a direction")
    if curve_a.link != curve_b.link:
        raise ValueError("plan_ratio_width: curve_a and curve_b must share a link")
    assert curve_a.params is not None and curve_b.params is not None
    for name in ("upper", "lower"):
        a, b = curve_a.params[name], curve_b.params[name]
        if a.fixed != b.fixed or (a.fixed and a.value != b.value):
            raise ValueError(
                f"plan_ratio_width: curve_a and curve_b must treat the {name} asymptote alike "
                "(both estimated, or both fixed at the same value)"
            )


def plan_ratio_width(
    curve_a: Fit,
    curve_b: Fit,
    *,
    severity: Sequence[float],
    trials_per_level: int,
    definition: Definition,
    interval_method: str,
    ratio_method: str,
    dependence: str,
    level: float,
    n_replicates: int,
    seed: int,
    n_clusters: int | None = None,
    icc: float = 0.0,
    n_resamples: int | None = None,
    workers: int = 1,
    provenance: Mapping[str, Any] | None = None,
) -> WidthPlan:
    """The widths of a threshold ratio's confidence interval to expect at a candidate design.

    As :func:`plan_threshold_width`, for two arms: each replicate draws data for both pilot
    curves at the same design, fits each, solves both thresholds with ``definition`` and
    ``interval_method``, and calls :func:`~robustmargin.ratio_interval` with ``ratio_method`` and
    ``dependence``. The width is ``log(hi / lo)`` of the ratio's interval. A replicate whose
    ratio is not ``OK``, is censored, is not a bounded interval (``UNBOUNDED``, ``EXCLUSIVE`` or
    ``HALF_OPEN``) or has a lower end that is not positive (``"LOWER_AT_ZERO"``) has no finite
    width and is tallied by reason; nothing is dropped. The ratio is computed from whatever
    thresholds the replicate gives, so a threshold with no finite end on either side can still
    give a finite ratio width.

    With ``dependence="independent"`` the arms are independent; given ``n_clusters`` each arm has
    its own ``n_clusters`` clusters, disjoint from the other's. With ``dependence="paired"`` the
    arms are observed on the same ``n_clusters`` clusters and share their effects, so the two
    outcomes of one cluster are correlated; this needs ``n_clusters``. The two arms use the
    pilot curves' own parameters, so a ratio of 1 is not assumed. See :func:`plan_threshold_width`
    for the process, reproducibility and cost.

    Parameters
    ----------
    curve_a, curve_b
        Pilot fits, both binomial and ``Status.OK``, on the same axis name, unit and scale, with
        the same direction, link and asymptote specification (each asymptote fixed at the same
        value in both, or estimated in both).
    severity, trials_per_level, definition, interval_method, level, n_replicates, seed, \
n_clusters, icc, n_resamples, workers, provenance
        As for :func:`plan_threshold_width`. ``trials_per_level`` is per arm.
    ratio_method
        Passed to :func:`~robustmargin.ratio_interval` (``"fieller"``, ``"log_delta"`` or
        ``"cluster_bootstrap"``). Ratio warnings are not stored on the result.
    dependence
        ``"independent"`` or ``"paired"``.

    Returns
    -------
    WidthPlan
        With ``kind="ratio"`` and ``width_scale="log_ratio"``.

    Raises
    ------
    TypeError
        If an integer argument is a ``bool`` or not an integer.
    ValueError
        As for :func:`plan_threshold_width`, and if the two curves differ in axis, direction,
        link or asymptote specification, ``ratio_method`` is not a ratio method, ``dependence``
        is not ``"independent"`` or ``"paired"``, or ``"paired"`` is given without ``n_clusters``.
    """
    _check_pilot(curve_a, name="curve_a")
    _check_pilot(curve_b, name="curve_b")
    _same_spec(curve_a, curve_b)
    if dependence not in _RATIO_DEPENDENCE:
        raise ValueError(
            f"dependence must be one of {sorted(_RATIO_DEPENDENCE)}, got {dependence!r}"
        )
    if dependence == "paired" and n_clusters is None:
        raise ValueError("dependence='paired' needs n_clusters")
    levels, icc_value, level_value = _check_design(
        severity,
        trials_per_level,
        n_clusters,
        icc,
        level,
        n_replicates,
        seed,
        workers,
        definition,
        n_resamples,
        interval_method,
        ratio_method,
    )
    spec = _Spec(
        curves=(curve_a, curve_b),
        severity=levels,
        trials_per_level=trials_per_level,
        definition=definition,
        interval_method=interval_method,
        ratio_method=ratio_method,
        dependence=dependence,
        level=level_value,
        n_clusters=n_clusters,
        icc=icc_value,
        n_resamples=n_resamples,
    )
    results = _execute(spec, n_replicates=n_replicates, seed=seed, workers=workers)
    return _build_plan(
        spec, results, kind="ratio", axis=curve_a.axis, seed=seed, provenance=provenance
    )
