"""Shared builders and independent oracles for the clustered-inference unit tests.

Nothing here imports robustmargin internals. The oracles are written from the mathematics, not
from the implementation: a per-trial Bernoulli log-likelihood on the natural ``(mu, s[, upper,
lower])`` scale, differentiated numerically, and a cluster sandwich assembled from it.

The curve is ``P(x) = u - (u - l) * F((t - mu) / s)`` with ``t = log x`` on a log axis (``x = 0``
gives ``P = u``) and ``F`` the link's inverse CDF; the outcome counted is the success.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import expit
from scipy.stats import norm

from robustmargin import Axis, Fit, Observations, Status
from robustmargin import intervals as intervals_module
from robustmargin import models as models_module
from robustmargin import ratio as ratio_module
from robustmargin.models import Parameter

LOG = Axis(name="severity", unit="unit", scale="log")
LINEAR = Axis(name="severity", unit="unit", scale="linear")

# (severity, cluster, successes, trials)
Row = tuple[float, str, int, int]

LEVELS: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
MU = math.log(6.0)
RICH_LEVELS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 32.0, 64.0)
SCALE = 0.6


def link_cdf(z: float | NDArray[np.float64], link: str) -> NDArray[np.float64]:
    z = np.asarray(z, dtype=np.float64)
    if link == "probit":
        return np.asarray(norm.cdf(z), dtype=np.float64)
    if link == "logit":
        return np.asarray(expit(z), dtype=np.float64)
    return np.asarray(-np.expm1(-np.exp(z)), dtype=np.float64)  # cloglog


def curve(
    severity: NDArray[np.float64],
    *,
    mu: float,
    s: float,
    upper: float,
    lower: float,
    link: str,
    scale: str = "log",
) -> NDArray[np.float64]:
    with np.errstate(divide="ignore"):
        t = np.log(severity) if scale == "log" else severity
    z = (t - mu) / s
    return upper - (upper - lower) * link_cdf(z, link)


def simulate_rows(
    seed: int,
    *,
    clusters: int,
    trials: int | Sequence[int] = 16,
    sigma: float = 0.8,
    levels: Sequence[float] = LEVELS,
    drop: dict[str, set[float]] | None = None,
    upper: float = 1.0,
    lower: float = 0.0,
    mu: float = MU,
    s: float = SCALE,
    scale: str = "log",
) -> list[Row]:
    """Probit-normal clustered data: cluster ``c`` has ``P_c(x) = 1 - Phi((t - mu)/s + b_c)``,
    ``b_c ~ N(0, sigma^2)``. ``trials`` is one count for every cluster or one per cluster;
    ``drop`` removes levels a cluster never observed."""
    rng = np.random.default_rng(seed)
    sizes = [trials] * clusters if isinstance(trials, int) else list(trials)
    rows: list[Row] = []
    x = np.asarray(levels, dtype=np.float64)
    for c in range(clusters):
        name = f"c{c:02d}"
        b = sigma * rng.standard_normal()
        with np.errstate(divide="ignore"):
            t = np.log(x) if scale == "log" else x
        p = lower + (upper - lower) * (1.0 - norm.cdf((t - mu) / s + b))
        for level, prob in zip(levels, p, strict=True):
            if drop is not None and level in drop.get(name, set()):
                continue
            n = sizes[c]
            rows.append((float(level), name, int(rng.binomial(n, prob)), n))
    return rows


def counts_obs(
    rows: Sequence[Row], *, direction: str = "decreasing", axis: Axis = LOG, clustered: bool = True
) -> Observations:
    return Observations.from_counts(
        axis,
        severity=[r[0] for r in rows],
        successes=[r[2] for r in rows],
        trials=[r[3] for r in rows],
        outcome="success",
        direction=direction,
        cluster=[r[1] for r in rows] if clustered else None,
    )


def trial_obs(rows: Sequence[Row], *, axis: Axis = LOG) -> Observations:
    """The same data one row per trial (``trials = 1``)."""
    severity: list[float] = []
    success: list[int] = []
    cluster: list[str] = []
    for sev, name, k, n in rows:
        severity += [sev] * n
        success += [1] * k + [0] * (n - k)
        cluster += [name] * n
    return Observations.from_observations(
        axis,
        severity=severity,
        success=success,
        outcome="success",
        direction="decreasing",
        cluster=cluster,
    )


def covariance_array(fit: Fit) -> NDArray[np.float64]:
    assert fit.covariance is not None
    return np.array(fit.covariance.matrix, dtype=np.float64)


def free_names(fit: Fit) -> list[str]:
    assert fit.params is not None
    return [name for name in ("mu", "s", "upper", "lower") if not fit.params[name].fixed]


def relative_matrix_error(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Largest entry difference, each scaled by the geometric mean of the two diagonals so that
    parameters of different units are judged on one footing."""
    scale = np.sqrt(np.outer(np.abs(np.diag(b)), np.abs(np.diag(b))))
    return float(np.max(np.abs(a - b) / scale))


# --------------------------------------------------------------------------------------------
# The oracle: per-trial Bernoulli log-likelihood, natural scale, numerical derivatives.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Oracle:
    scores: NDArray[np.float64]
    cluster_ids: tuple[str, ...]
    covariance: NDArray[np.float64]
    meat_unscaled: NDArray[np.float64]
    bread: NDArray[np.float64]
    clusters: int


def _loglik(
    theta: NDArray[np.float64],
    names: Sequence[str],
    fixed: dict[str, float],
    rows: Sequence[Row],
    *,
    link: str,
    scale: str,
) -> float:
    values = dict(fixed)
    values.update(zip(names, theta, strict=True))
    severity = np.array([r[0] for r in rows], dtype=np.float64)
    k = np.array([r[2] for r in rows], dtype=np.float64)
    n = np.array([r[3] for r in rows], dtype=np.float64)
    p = curve(
        severity,
        mu=values["mu"],
        s=values["s"],
        upper=values["upper"],
        lower=values["lower"],
        link=link,
        scale=scale,
    )
    p = np.clip(p, 1e-300, 1.0)
    q = np.clip(1.0 - p, 1e-300, 1.0)
    return float(np.sum(k * np.log(p) + (n - k) * np.log(q)))


def _gradient(f, theta: NDArray[np.float64]) -> NDArray[np.float64]:  # type: ignore[no-untyped-def]
    grad = np.zeros_like(theta)
    for i in range(theta.size):
        h = 1e-5 * max(1.0, abs(theta[i]))
        up = theta.copy()
        down = theta.copy()
        up[i] += h
        down[i] -= h
        grad[i] = (f(up) - f(down)) / (2.0 * h)
    return grad


def _hessian(f, theta: NDArray[np.float64]) -> NDArray[np.float64]:  # type: ignore[no-untyped-def]
    size = theta.size
    hess = np.zeros((size, size))
    for i in range(size):
        h = 1e-4 * max(1.0, abs(theta[i]))
        up = theta.copy()
        down = theta.copy()
        up[i] += h
        down[i] -= h
        hess[i] = (_gradient(f, up) - _gradient(f, down)) / (2.0 * h)
    return (hess + hess.T) / 2.0


def oracle_sandwich(fit: Fit, rows: Sequence[Row], *, link: str | None = None) -> Oracle:
    """The cluster sandwich ``B M B`` at the fit's own natural-scale estimates, with
    ``B = (-Hessian)^-1`` of the total per-trial log-likelihood, ``M = G/(G-1) sum_c s_c s_c'``
    and ``s_c`` the gradient of cluster ``c``'s own log-likelihood. The step sizes give about
    seven correct digits."""
    assert fit.params is not None
    link = link or fit.link
    scale = fit.axis.scale
    assert fit.covariance is not None
    names = list(fit.covariance.names)
    fixed = {name: p.value for name, p in fit.params.items() if p.fixed}
    theta = np.array([fit.params[name].value for name in names], dtype=np.float64)

    def total(th: NDArray[np.float64]) -> float:
        return _loglik(th, names, fixed, rows, link=link, scale=scale)

    ids = list(dict.fromkeys(r[1] for r in rows))
    scores = []
    for c in ids:
        mine = [r for r in rows if r[1] == c]

        def own(th: NDArray[np.float64], mine: list[Row] = mine) -> float:
            return _loglik(th, names, fixed, mine, link=link, scale=scale)

        scores.append(_gradient(own, theta))
    s = np.array(scores)
    bread = np.asarray(np.linalg.inv(-_hessian(total, theta)), dtype=np.float64)
    g = len(ids)
    meat0 = s.T @ s
    covariance = bread @ (g / (g - 1.0) * meat0) @ bread
    return Oracle(
        scores=s,
        cluster_ids=tuple(ids),
        covariance=covariance,
        meat_unscaled=meat0,
        bread=bread,
        clusters=g,
    )


def simulate_paired_rows(
    seed: int,
    *,
    clusters: int,
    rho: float,
    trials: int = 16,
    sigma: float = 0.8,
    levels_a: Sequence[float] = LEVELS,
    levels_b: Sequence[float] = LEVELS,
    mu_a: float = MU,
    mu_b: float = MU,
    s_a: float = SCALE,
    s_b: float = SCALE,
) -> tuple[list[Row], list[Row]]:
    """Two outcomes measured on the same clusters. The cluster effects of the two outcomes are
    ``N(0, sigma^2)`` each, with correlation ``rho``."""
    rng = np.random.default_rng(seed)
    out: tuple[list[Row], list[Row]] = ([], [])
    for c in range(clusters):
        name = f"c{c:02d}"
        shared = rng.standard_normal()
        own = rng.standard_normal()
        effects = (sigma * shared, sigma * (rho * shared + math.sqrt(1.0 - rho**2) * own))
        for rows, b, levels, mu, s in zip(
            out, effects, (levels_a, levels_b), (mu_a, mu_b), (s_a, s_b), strict=True
        ):
            x = np.asarray(levels, dtype=np.float64)
            with np.errstate(divide="ignore"):
                t = np.log(x)
            p = 1.0 - norm.cdf((t - mu) / s + b)
            for level, prob in zip(levels, p, strict=True):
                rows.append((float(level), name, int(rng.binomial(trials, prob)), trials))
    return out


# --------------------------------------------------------------------------------------------
# Scripted replicates: the refit entry is wrapped so replicate i's parameters are chosen.
# --------------------------------------------------------------------------------------------

Script = Callable[[int], tuple[float, float, float]]  # replicate index -> (mu, upper, lower)


def install_scripts(monkeypatch: pytest.MonkeyPatch, scripts: dict[str, Script]) -> None:
    """Replace each refit of an outcome named in ``scripts`` by its scripted parameters."""
    real = models_module._fit_counts
    counters: defaultdict[str, int] = defaultdict(int)

    def scripted(obs: Observations, **kwargs: Any) -> Fit:
        fit = real(obs, **kwargs)
        script = scripts.get(obs.outcome)
        if script is None or fit.status is not Status.OK or fit.params is None:
            return fit
        index = counters[obs.outcome]
        counters[obs.outcome] += 1
        mu, upper, lower = script(index)
        params = dict(fit.params)
        params["mu"] = Parameter(value=mu, fixed=False)
        params["upper"] = Parameter(value=upper, fixed=True)
        params["lower"] = Parameter(value=lower, fixed=True)
        return replace(fit, params=params)

    for module in (models_module, intervals_module, ratio_module):
        monkeypatch.setattr(module, "_fit_counts", scripted, raising=False)


def finite(mu: float) -> tuple[float, float, float]:
    return mu, 1.0, 0.0


HIGH = (0.0, 0.4, 0.0)  # target 0.5 >= upper: theta = -inf  (unreachable on the low side)
LOW = (0.0, 1.0, 0.55)  # target 0.5 <= lower: theta = +inf  (unreachable on the high side)
