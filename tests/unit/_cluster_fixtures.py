"""Shared fixtures for the clustered and paired API-surface tests.

Everything is built through the public API only, from hand-written deterministic counts (no
random source, so no seed to pin). The designs have G clusters observed at every level of a
log-spaced severity grid with an all-success control, so a fixed-asymptote fit has two estimated
parameters and G > 2.
"""

from __future__ import annotations

from robustmargin import (
    Axis,
    Definition,
    Fit,
    Observations,
    Threshold,
    fit_dose_response,
    threshold,
)

AXIS = Axis(name="axis_1", unit="m", scale="log")
LEVELS: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
TRIALS_PER_CLUSTER = 10
# Successes out of TRIALS_PER_CLUSTER at each level for the baseline cluster; each cluster adds
# an offset that depends on both the cluster and the level (clipped to the valid range). A
# cluster-only offset makes every cluster's score vector a multiple of one vector, so the cluster
# sandwich is exactly singular; the level term gives the score matrix full rank.
_BASE: dict[str, tuple[int, ...]] = {
    "a": (10, 9, 8, 6, 3, 1),
    "b": (10, 8, 6, 4, 2, 0),
}
_OFFSETS: tuple[int, ...] = (-1, 0, 1, 0, -1, 1, 0, 1, -1, 0)


def cluster_names(g: int, prefix: str = "c") -> tuple[str, ...]:
    return tuple(f"{prefix}{i}" for i in range(g))


def clustered_counts(
    g: int = 6,
    *,
    prefix: str = "c",
    outcome: str = "success",
    base: str = "a",
    direction: str = "decreasing",
) -> Observations:
    """``g`` clusters x every level, aggregated rows with a cluster id per row."""
    severity: list[float] = []
    successes: list[int] = []
    trials: list[int] = []
    cluster: list[str] = []
    for index, name in enumerate(cluster_names(g, prefix)):
        for level_index, (level, rate) in enumerate(zip(LEVELS, _BASE[base], strict=True)):
            offset = _OFFSETS[(index + 2 * level_index) % len(_OFFSETS)]
            k = rate if level == 0.0 else min(TRIALS_PER_CLUSTER, max(0, rate + offset))
            severity.append(level)
            successes.append(k)
            trials.append(TRIALS_PER_CLUSTER)
            cluster.append(name)
    if direction == "increasing":
        successes = [TRIALS_PER_CLUSTER - k for k in successes]
    return Observations.from_counts(
        AXIS,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome=outcome,
        direction=direction,
        cluster=cluster,
    )


def clustered_fit(
    g: int = 6, *, prefix: str = "c", base: str = "a", dependence: str = "clustered"
) -> Fit:
    return fit_dose_response(
        clustered_counts(g, prefix=prefix, base=base),
        model="binomial",
        link="probit",
        upper=1.0,
        lower=0.0,
        dependence=dependence,
    )


def clustered_threshold(
    fit: Fit | None = None,
    *,
    target: float = 0.5,
    interval_method: str = "delta",
    n_resamples: int | None = None,
    seed: int | None = None,
) -> Threshold:
    return threshold(
        fit if fit is not None else clustered_fit(),
        definition=Definition.absolute(target),
        interval_method=interval_method,
        level=0.95,
        dependence="clustered",
        n_resamples=n_resamples,
        seed=seed,
    )


def paired_fits(g: int = 6) -> tuple[Fit, Fit]:
    """Two outcomes measured on the same clusters."""
    return clustered_fit(g, base="a"), clustered_fit(g, base="b")
