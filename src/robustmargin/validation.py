"""Fit diagnostics (model-only) and the data-only adjacent-level monotonicity check.

Two independent things live here, deliberately kept apart:

- :class:`Diagnostics` / :func:`diagnose` need a converged fit -- goodness of fit, dispersion,
  and link comparison are all statements about how well a fitted curve matches the data, so they
  are ``None`` whenever :attr:`~robustmargin.Fit.status` is not :data:`~robustmargin.Status.OK`
  (a failed fit carries no model-derived numbers).
- :class:`MonotonicityCheck` / :func:`check_monotonicity` need only raw per-level counts -- an
  exact one-sided Fisher test on adjacent severity levels, Holm-corrected across the family. It
  therefore runs on *every* :func:`~robustmargin.fit_dose_response` return path, whatever the
  ``Status``, because non-monotone data is a common cause of a failed fit and disappearing the
  check exactly then would hide the finding it exists to surface.

Both are reported and never acted on automatically: nothing here changes a
reported number, chooses a link, or corrects for dispersion. ``Fit.link`` stays whatever the
caller passed.

A Gaussian (continuous-outcome) fit has the same two pieces in their own form:
:class:`ContinuousDiagnostics` (residual sum of squares, a pure-error lack-of-fit F test when
the design has replicates, and the link comparison by AIC) and, for the monotonicity check, a
one-sided Welch t-test between adjacent levels (:class:`ContinuousAdjacentPair`), Holm-corrected
across the family. :func:`diagnose` and :func:`check_monotonicity` dispatch on the type of what
they are given.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import xlogy
from scipy.stats import f as f_distribution
from scipy.stats import fisher_exact
from scipy.stats import t as t_distribution
from statsmodels.stats.multitest import multipletests

from robustmargin.types import (
    Cell,
    ContinuousCell,
    ContinuousObservations,
    Observations,
    Status,
    _check_finite_number,
)

if TYPE_CHECKING:
    # Only for the `diagnose(fit: Fit)` annotation. `from __future__ import annotations` (above)
    # makes every annotation in this module a string, so this import is never evaluated at
    # runtime -- it exists purely for mypy, and keeps this module from ever importing
    # robustmargin.models at import time (models.py imports *this* module at its own top level, to
    # build Fit.diagnostics/Fit.monotonicity and to call diagnose()/check_monotonicity() from
    # fit_dose_response; the reverse, real import would be a cycle).
    from robustmargin.models import Fit

__all__ = [
    "AdjacentPair",
    "ContinuousAdjacentPair",
    "ContinuousDiagnostics",
    "Diagnostics",
    "MonotonicityCheck",
    "check_monotonicity",
    "diagnose",
]

# A warning when phi > 1.5 -- strict, an exact 1.5 does not warn.
_DISPERSION_WARNING_THRESHOLD = 1.5
_OVERDISPERSED_WARNING_PREFIX = "OVERDISPERSED: "
_SPARSE_CELLS_WARNING_PREFIX = "SPARSE_CELLS: "
# A counted cell is flagged on one side (successes or failures) only when *both* hold -- the
# expected count on that side is below this cutoff, and the *observed* count on that side is at
# least 1. A small expected count with nothing observed there is a clean plateau, not a finding;
# an observed event against a merely-moderate expectation (down to about 0.43 near a design's
# transition) is ordinary too -- only a genuine event against an expectation this small is the
# "dominated by a single event" pathology the warning exists for. Report-only: nothing here is
# recomputed. A fixed constant, not a parameter; changing it is a statistical-method change.
_SPARSE_EXPECTED_CUTOFF = 0.1
# R's own `binomial()$linkinv` clamp (`.Machine$double.eps`).
_PROBABILITY_EPS = float(np.finfo(float).eps)

# Link comparison covers exactly these three binomial links.
_COMPARISON_LINKS: tuple[str, ...] = ("probit", "logit", "cloglog")

_MONOTONICITY_METHOD = "fisher_exact_one_sided_holm"
_CONTINUOUS_MONOTONICITY_METHOD = "welch_t_one_sided_holm"
_MONOTONICITY_METHODS = frozenset({_MONOTONICITY_METHOD, _CONTINUOUS_MONOTONICITY_METHOD})
_VALID_DIRECTIONS = frozenset({"decreasing", "increasing"})
_VALID_DEPENDENCE_INPUTS = frozenset({"independent"})


def _check_link_tables(
    link_aic: Mapping[str, float | None],
    link_status: Mapping[str, Status],
    *,
    label: str,
) -> None:
    """The invariants shared by :class:`Diagnostics` and :class:`ContinuousDiagnostics` on their
    link comparison: exactly the three links, a ``Status`` for each, and an AIC present exactly
    when that link's fit is ``OK``."""
    if set(link_aic) != set(_COMPARISON_LINKS):
        raise ValueError(
            f"{label}.link_aic must have exactly the keys {sorted(_COMPARISON_LINKS)}, "
            f"got {sorted(link_aic)}"
        )
    if set(link_status) != set(_COMPARISON_LINKS):
        raise ValueError(
            f"{label}.link_status must have exactly the keys "
            f"{sorted(_COMPARISON_LINKS)}, got {sorted(link_status)}"
        )
    for link in _COMPARISON_LINKS:
        aic = link_aic[link]
        status = link_status[link]
        if not isinstance(status, Status):
            raise ValueError(f"{label}.link_status[{link!r}] must be a Status, got {status!r}")
        if aic is None:
            if status is Status.OK:
                raise ValueError(
                    f"{label}.link_aic[{link!r}] is None, but "
                    f"link_status[{link!r}] is Status.OK -- an OK refit always has an AIC"
                )
        else:
            _check_finite_number(aic, label=f"{label}.link_aic[{link!r}]")
            if status is not Status.OK:
                raise ValueError(
                    f"{label}.link_aic[{link!r}] is set, but link_status[{link!r}] is "
                    f"{status!r}, not Status.OK"
                )


@dataclass(frozen=True, kw_only=True)
class Diagnostics:
    """Model-only fit diagnostics: goodness of fit, dispersion, and link comparison. The
    monotonicity check is not part of this type -- see :class:`MonotonicityCheck`.

    Built only from a converged fit (:func:`diagnose` requires ``fit.status is Status.OK``), and
    attached to :attr:`~robustmargin.Fit.diagnostics` only in that case: ``None`` whenever the fit
    failed (a failed fit carries no model-derived numbers). Contrast :class:`MonotonicityCheck`,
    attached to :attr:`~robustmargin.Fit.monotonicity` on *every* ``fit_dose_response`` return path
    regardless of ``Status`` -- the two fields on ``Fit`` have different nullability for exactly
    this reason.

    Attributes
    ----------
    deviance
        The binomial deviance on the per-severity-level cells, or ``None`` when ``df <= 0``
        (see ``df`` below) -- not a failed fit, just no goodness-of-fit statistic to report. A
        cell is excluded from this sum only when it is *structurally* deterministic under the
        model -- a log-axis cell at severity exactly ``0.0`` when ``upper`` is *fixed* at exactly
        ``1.0`` (``direction="decreasing"``, or the equivalent ``lower`` fixed at exactly ``0.0``
        for an increasing outcome); that cell has zero variance and zero deviance by
        construction. Every other cell counts, including one whose fitted probability happens to
        round to exactly ``0`` or ``1`` in double precision: probabilities are computed to avoid
        needless cancellation and clamped to ``[eps, 1 - eps]`` (``eps`` the machine epsilon,
        matching R's own ``binomial()$linkinv``), so such a cell contributes approximately zero
        rather than a ``0/0``. The exclusion is by structure, not by *computed* value, so ``df``
        does not depend on the link.
    pearson_chi2
        The Pearson chi-square statistic on the same cells, with the same exclusion and the
        same ``None``-on-``df <= 0`` rule as ``deviance``.
    df
        Degrees of freedom: the number of cells counted towards ``deviance``/``pearson_chi2``
        (after the structural exclusion above) minus the number of estimated parameters. Always
        an integer, even when it is zero or negative.
    dispersion
        ``pearson_chi2 / df`` (``phi``), or ``None`` exactly when ``pearson_chi2`` is ``None``
        (``df <= 0``). A value strictly greater than ``1.5`` adds a warning starting with
        ``"OVERDISPERSED: "``; nothing is rescaled and no interval changes as a result (reported,
        never acted on). It is a screen, not a test: under a correctly specified model
        ``phi > 1.5`` fires about 21% of the time at ``df`` 1 to 3 and about 15% at ``df`` 8.
        ``phi`` can be dominated by a single observed event against a tiny expected
        count -- see ``warnings``' ``"SPARSE_CELLS: "`` entry, which flags exactly that case without
        changing ``dispersion`` itself.
    link_aic
        AIC for each of ``"probit"``, ``"logit"`` and ``"cloglog"``, refit under the same
        ``upper``/``lower`` specification as the diagnosed fit. The diagnosed fit's own link
        reuses its own ``log_likelihood`` and parameter count directly (never refit, so it is
        exact by construction); the other two links are refit through the existing statsmodels
        fitting paths (never a hand-rolled fitter). ``None`` for a link whose refit does not
        reach :data:`~robustmargin.Status.OK` -- see ``link_status``.
    link_status
        The :class:`~robustmargin.Status` of each link's own fit (the diagnosed fit's own status,
        which is always ``OK``, for its own link; the refit's status for the other two).
        ``link_aic[link] is None`` if and only if ``link_status[link] is not Status.OK``.
    warnings
        Free-text notes: the ``df <= 0`` reason, an ``"OVERDISPERSED: "`` token, or a
        ``"SPARSE_CELLS: "`` token naming a counted cell flagged on one side (successes or
        failures) because that side's expected count is below ``0.1`` *and* at least one event
        was actually observed there -- report-only, like ``"OVERDISPERSED: "``: it changes no
        number, only says that ``pearson_chi2`` and ``dispersion`` are unreliable at that cell.
        Empty by default.
    """

    deviance: float | None
    pearson_chi2: float | None
    df: int
    dispersion: float | None
    link_aic: Mapping[str, float | None]
    link_status: Mapping[str, Status]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "link_aic", dict(self.link_aic))
        object.__setattr__(self, "link_status", dict(self.link_status))

        if isinstance(self.df, bool) or not isinstance(self.df, int):
            raise ValueError(f"Diagnostics.df must be a genuine int, got {self.df!r}")

        gof_fields = (self.deviance, self.pearson_chi2, self.dispersion)
        gof_all_none = all(v is None for v in gof_fields)
        gof_all_set = all(v is not None for v in gof_fields)
        if not (gof_all_none or gof_all_set):
            raise ValueError(
                "Diagnostics.deviance, .pearson_chi2 and .dispersion must be either all None "
                f"(df <= 0) or all set, got deviance={self.deviance!r}, "
                f"pearson_chi2={self.pearson_chi2!r}, dispersion={self.dispersion!r}"
            )
        if gof_all_none and self.df > 0:
            raise ValueError(
                "Diagnostics.deviance/.pearson_chi2/.dispersion are None, but df "
                f"({self.df!r}) is > 0 -- they are only None when df <= 0"
            )
        if gof_all_set:
            assert self.deviance is not None and self.pearson_chi2 is not None
            assert self.dispersion is not None
            _check_finite_number(self.deviance, label="Diagnostics.deviance")
            _check_finite_number(self.pearson_chi2, label="Diagnostics.pearson_chi2")
            _check_finite_number(self.dispersion, label="Diagnostics.dispersion")
            if self.df <= 0:
                raise ValueError(
                    f"Diagnostics.df is {self.df!r} (<= 0), so deviance/pearson_chi2/dispersion "
                    "must be None"
                )

        _check_link_tables(self.link_aic, self.link_status, label="Diagnostics")


@dataclass(frozen=True, kw_only=True)
class ContinuousDiagnostics:
    """Model-only diagnostics for a Gaussian fit: the residual sum of squares, a pure-error
    lack-of-fit F test, and the comparison of the three links by AIC.

    Built only from a converged fit and attached to :attr:`~robustmargin.Fit.diagnostics` only in
    that case. Reported and never acted on: nothing here rescales an interval or picks a link.

    Attributes
    ----------
    rss
        The residual sum of squares over every observation, equal to ``Fit.residual.rss``.
    df
        The residual degrees of freedom, equal to ``Fit.residual.df``.
    pure_error_ss
        The within-level sum of squares: the part of ``rss`` that no curve can remove.
    pure_error_df
        The pure-error degrees of freedom, the number of observations minus the number of levels.
        Zero when no level is replicated.
    lack_of_fit_f
        The lack-of-fit F statistic, ``((rss - pure_error_ss) / lof_df) / (pure_error_ss /
        pure_error_df)`` with ``lof_df = df - pure_error_df`` (the number of levels minus the
        number of estimated curve parameters). ``None`` when there is no replication, no
        degrees of freedom for lack of fit, or no spread within the levels.
    lack_of_fit_p
        The upper-tail ``F(lof_df, pure_error_df)`` probability of ``lack_of_fit_f``; ``None``
        exactly when ``lack_of_fit_f`` is. A small value says the curve misses the level means by
        more than the replicate scatter explains. It is a screen, and changes nothing.
    link_aic
        AIC for each of ``"probit"``, ``"logit"`` and ``"cloglog"``, refit under the same
        ``upper``/``lower`` specification. The residual variance counts as one more parameter
        for every link, so it does not affect the comparison. ``None`` for a link whose refit does
        not reach :data:`~robustmargin.Status.OK`.
    link_status
        The :class:`~robustmargin.Status` of each link's own fit. ``link_aic[link] is None`` if
        and only if ``link_status[link]`` is not ``OK``.
    warnings
        Free-text notes. Empty by default.

    Raises
    ------
    ValueError
        If a number is not finite, ``df`` is not a positive int, ``pure_error_df`` is negative,
        ``lack_of_fit_f`` and ``lack_of_fit_p`` are not both set or both ``None``, or the link
        tables do not cover exactly the three links consistently.
    """

    rss: float
    df: int
    pure_error_ss: float
    pure_error_df: int
    lack_of_fit_f: float | None
    lack_of_fit_p: float | None
    link_aic: Mapping[str, float | None]
    link_status: Mapping[str, Status]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "link_aic", dict(self.link_aic))
        object.__setattr__(self, "link_status", dict(self.link_status))

        for label, value in (("rss", self.rss), ("pure_error_ss", self.pure_error_ss)):
            _check_finite_number(value, label=f"ContinuousDiagnostics.{label}")
            if value < 0.0:
                raise ValueError(
                    f"ContinuousDiagnostics.{label} must be non-negative, got {value!r}"
                )
        for label, count in (("df", self.df), ("pure_error_df", self.pure_error_df)):
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError(
                    f"ContinuousDiagnostics.{label} must be a genuine int, got {count!r}"
                )
        if self.df < 1:
            raise ValueError(f"ContinuousDiagnostics.df must be >= 1, got {self.df!r}")
        if self.pure_error_df < 0:
            raise ValueError(
                f"ContinuousDiagnostics.pure_error_df must be >= 0, got {self.pure_error_df!r}"
            )
        if (self.lack_of_fit_f is None) != (self.lack_of_fit_p is None):
            raise ValueError(
                "ContinuousDiagnostics.lack_of_fit_f and .lack_of_fit_p must be both set or both "
                f"None, got {self.lack_of_fit_f!r} and {self.lack_of_fit_p!r}"
            )
        if self.lack_of_fit_f is not None and self.lack_of_fit_p is not None:
            _check_finite_number(self.lack_of_fit_f, label="ContinuousDiagnostics.lack_of_fit_f")
            _check_finite_number(self.lack_of_fit_p, label="ContinuousDiagnostics.lack_of_fit_p")
            if self.lack_of_fit_f < 0.0 or not (0.0 <= self.lack_of_fit_p <= 1.0):
                raise ValueError(
                    "ContinuousDiagnostics.lack_of_fit_f must be non-negative and "
                    f"lack_of_fit_p must lie in [0, 1], got {self.lack_of_fit_f!r}, "
                    f"{self.lack_of_fit_p!r}"
                )
        _check_link_tables(self.link_aic, self.link_status, label="ContinuousDiagnostics")


@dataclass(frozen=True, kw_only=True)
class AdjacentPair:
    """One adjacent-severity-level pair from a :class:`MonotonicityCheck`: the raw counts at
    both levels, the one-sided Fisher exact test between them, and its Holm-adjusted p-value.

    Attributes
    ----------
    severity_low, severity_high
        The two adjacent severity levels, ``severity_low < severity_high``.
    successes_low, trials_low, successes_high, trials_high
        The raw pooled counts at each level (:meth:`~robustmargin.Observations.cells`).
    p_value
        The raw one-sided Fisher exact p-value for a reversal against ``direction`` (that is, the
        higher severity performing *better* than the lower one, which is evidence against
        ``direction="decreasing"``; the mirror image against ``"increasing"``) --
        :func:`scipy.stats.fisher_exact` on
        ``[[successes_high, failures_high], [successes_low, failures_low]]``,
        ``alternative="greater"`` for ``direction="decreasing"``, ``"less"`` for
        ``"increasing"``.
    holm_adjusted_p
        ``p_value`` after a Holm step-down correction across every pair in the same
        :class:`MonotonicityCheck` (:func:`statsmodels.stats.multitest.multipletests`,
        ``method="holm"``).
    flagged
        ``holm_adjusted_p < alpha`` (``alpha`` recorded on the enclosing
        :class:`MonotonicityCheck`).
    """

    severity_low: float
    severity_high: float
    successes_low: int
    trials_low: int
    successes_high: int
    trials_high: int
    p_value: float
    holm_adjusted_p: float
    flagged: bool

    def __post_init__(self) -> None:
        _check_finite_number(self.severity_low, label="AdjacentPair.severity_low")
        _check_finite_number(self.severity_high, label="AdjacentPair.severity_high")
        if not self.severity_low < self.severity_high:
            raise ValueError(
                "AdjacentPair.severity_low must be strictly less than .severity_high, got "
                f"severity_low={self.severity_low!r}, severity_high={self.severity_high!r}"
            )
        for label, value in (
            ("successes_low", self.successes_low),
            ("trials_low", self.trials_low),
            ("successes_high", self.successes_high),
            ("trials_high", self.trials_high),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"AdjacentPair.{label} must be a genuine int, got {value!r}")
        if not (0 <= self.successes_low <= self.trials_low):
            raise ValueError(
                "AdjacentPair requires 0 <= successes_low <= trials_low, got "
                f"successes_low={self.successes_low!r}, trials_low={self.trials_low!r}"
            )
        if not (0 <= self.successes_high <= self.trials_high):
            raise ValueError(
                "AdjacentPair requires 0 <= successes_high <= trials_high, got "
                f"successes_high={self.successes_high!r}, trials_high={self.trials_high!r}"
            )
        _check_finite_number(self.p_value, label="AdjacentPair.p_value")
        _check_finite_number(self.holm_adjusted_p, label="AdjacentPair.holm_adjusted_p")
        if not (0.0 <= self.p_value <= 1.0):
            raise ValueError(f"AdjacentPair.p_value must lie in [0, 1], got {self.p_value!r}")
        if not (0.0 <= self.holm_adjusted_p <= 1.0):
            raise ValueError(
                f"AdjacentPair.holm_adjusted_p must lie in [0, 1], got {self.holm_adjusted_p!r}"
            )
        if not isinstance(self.flagged, bool):
            raise ValueError(f"AdjacentPair.flagged must be a genuine bool, got {self.flagged!r}")


@dataclass(frozen=True, kw_only=True)
class ContinuousAdjacentPair:
    """One adjacent-severity-level pair from a Gaussian :class:`MonotonicityCheck`: each
    level's count, mean and sample SD, the one-sided Welch t-test between them, and its
    Holm-adjusted p-value.

    Attributes
    ----------
    severity_low, severity_high
        The two adjacent severity levels, ``severity_low < severity_high``.
    n_low, mean_low, sd_low, n_high, mean_high, sd_high
        The pooled count, mean and sample standard deviation (``ddof=1``) at each level.
    statistic
        The Welch t statistic ``(mean_high - mean_low) / sqrt(sd_low^2 / n_low + sd_high^2 /
        n_high)``, or ``None`` when the pair cannot be tested (see ``p_value``).
    df
        The Welch-Satterthwaite degrees of freedom, or ``None`` exactly when ``statistic`` is.
    p_value
        The one-sided p-value for a reversal against ``direction``: the higher level having the
        larger mean for a decreasing outcome (``alternative="greater"``), the smaller mean for an
        increasing one. A pair is untestable, and its p-value is recorded as ``1.0``, when either
        level has one observation or both levels have zero spread (no spread to measure).
    holm_adjusted_p
        ``p_value`` after a Holm step-down correction across every pair of the check.
    flagged
        ``holm_adjusted_p < alpha`` (``alpha`` on the enclosing :class:`MonotonicityCheck`).

    Raises
    ------
    ValueError
        If the levels are not strictly ordered, a count is not a positive int, a number is not
        finite, a standard deviation or p-value is out of range, or ``statistic`` and ``df`` are
        not both set or both ``None``.
    """

    severity_low: float
    severity_high: float
    n_low: int
    mean_low: float
    sd_low: float
    n_high: int
    mean_high: float
    sd_high: float
    statistic: float | None
    df: float | None
    p_value: float
    holm_adjusted_p: float
    flagged: bool

    def __post_init__(self) -> None:
        _check_finite_number(self.severity_low, label="ContinuousAdjacentPair.severity_low")
        _check_finite_number(self.severity_high, label="ContinuousAdjacentPair.severity_high")
        if not self.severity_low < self.severity_high:
            raise ValueError(
                "ContinuousAdjacentPair.severity_low must be strictly less than .severity_high, "
                f"got severity_low={self.severity_low!r}, severity_high={self.severity_high!r}"
            )
        for label, count in (("n_low", self.n_low), ("n_high", self.n_high)):
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(
                    f"ContinuousAdjacentPair.{label} must be a genuine int >= 1, got {count!r}"
                )
        for label, value in (
            ("mean_low", self.mean_low),
            ("mean_high", self.mean_high),
            ("sd_low", self.sd_low),
            ("sd_high", self.sd_high),
            ("p_value", self.p_value),
            ("holm_adjusted_p", self.holm_adjusted_p),
        ):
            _check_finite_number(value, label=f"ContinuousAdjacentPair.{label}")
        if self.sd_low < 0.0 or self.sd_high < 0.0:
            raise ValueError("ContinuousAdjacentPair.sd_low and .sd_high must be non-negative")
        if (self.statistic is None) != (self.df is None):
            raise ValueError(
                "ContinuousAdjacentPair.statistic and .df must be both set or both None, got "
                f"{self.statistic!r} and {self.df!r}"
            )
        if self.statistic is not None and self.df is not None:
            _check_finite_number(self.statistic, label="ContinuousAdjacentPair.statistic")
            _check_finite_number(self.df, label="ContinuousAdjacentPair.df")
            if not self.df > 0.0:
                raise ValueError(f"ContinuousAdjacentPair.df must be positive, got {self.df!r}")
        if not (0.0 <= self.p_value <= 1.0):
            raise ValueError(
                f"ContinuousAdjacentPair.p_value must lie in [0, 1], got {self.p_value!r}"
            )
        if not (0.0 <= self.holm_adjusted_p <= 1.0):
            raise ValueError(
                "ContinuousAdjacentPair.holm_adjusted_p must lie in [0, 1], got "
                f"{self.holm_adjusted_p!r}"
            )
        if not isinstance(self.flagged, bool):
            raise ValueError(
                f"ContinuousAdjacentPair.flagged must be a genuine bool, got {self.flagged!r}"
            )


@dataclass(frozen=True, kw_only=True)
class MonotonicityCheck:
    """A data-only adjacent-level monotonicity check: for each pair of adjacent severity levels
    in the data, a one-sided Fisher exact test for "the higher severity performs better" under
    ``direction``, Holm-corrected across the whole family.

    Needs only raw counts, never a fitted curve, so it is attached to
    :attr:`~robustmargin.Fit.monotonicity` on *every* :func:`~robustmargin.fit_dose_response` return
    path, whatever the resulting :class:`~robustmargin.Status` -- contrast :class:`Diagnostics`,
    which needs a converged fit and is ``None`` otherwise. See :func:`check_monotonicity`.

    Attributes
    ----------
    direction
        ``"decreasing"`` or ``"increasing"``, read from the ``Observations`` the check was built
        from (never passed separately, so the check cannot disagree with its own data).
    alpha
        The significance level ``flagged`` is decided at, strictly between 0 and 1.
    dependence
        ``"independent"`` in this release: the check assumes independent trials. Recorded so a
        serialised result states the assumption it relies on.
    method
        Names the statistical method: ``"fisher_exact_one_sided_holm"`` for counts and
        ``"welch_t_one_sided_holm"`` for a Gaussian series.
    pairs
        One :class:`AdjacentPair` (counts) or :class:`ContinuousAdjacentPair` (Gaussian series) per
        adjacent pair of severity levels, ascending. Empty when fewer than two distinct severity
        levels exist (``warnings`` then names the reason).
    any_flagged
        ``True`` exactly when at least one entry of ``pairs`` is ``flagged``.
    warnings
        Free-text notes (for example, naming a degenerate single-level input). Empty by default.
    """

    direction: str
    alpha: float
    dependence: str
    pairs: tuple[AdjacentPair | ContinuousAdjacentPair, ...]
    any_flagged: bool
    method: str = _MONOTONICITY_METHOD
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairs", tuple(self.pairs))
        object.__setattr__(self, "warnings", tuple(self.warnings))

        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"MonotonicityCheck.direction must be one of {sorted(_VALID_DIRECTIONS)}, "
                f"got {self.direction!r}"
            )
        _check_finite_number(self.alpha, label="MonotonicityCheck.alpha")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(
                f"MonotonicityCheck.alpha must satisfy 0 < alpha < 1, got {self.alpha!r}"
            )
        if self.dependence not in _VALID_DEPENDENCE_INPUTS:
            raise ValueError(
                "MonotonicityCheck.dependence must be one of "
                f"{sorted(_VALID_DEPENDENCE_INPUTS)}, got {self.dependence!r}"
            )
        if self.method not in _MONOTONICITY_METHODS:
            raise ValueError(
                f"MonotonicityCheck.method must be one of {sorted(_MONOTONICITY_METHODS)}, "
                f"got {self.method!r}"
            )
        if not isinstance(self.any_flagged, bool):
            raise ValueError(
                f"MonotonicityCheck.any_flagged must be a genuine bool, got {self.any_flagged!r}"
            )
        pair_type = (
            ContinuousAdjacentPair
            if self.method == _CONTINUOUS_MONOTONICITY_METHOD
            else AdjacentPair
        )
        for pair in self.pairs:
            if not isinstance(pair, pair_type):
                raise ValueError(
                    f"MonotonicityCheck.pairs entries must each be a {pair_type.__name__} for "
                    f"method={self.method!r}, got {pair!r}"
                )
        expected_any_flagged = any(pair.flagged for pair in self.pairs)
        if self.any_flagged != expected_any_flagged:
            raise ValueError(
                f"MonotonicityCheck.any_flagged ({self.any_flagged!r}) must equal "
                f"any(pair.flagged for pair in pairs) ({expected_any_flagged!r})"
            )
        for pair in self.pairs:
            expected_flagged = pair.holm_adjusted_p < self.alpha
            if pair.flagged != expected_flagged:
                raise ValueError(
                    f"AdjacentPair.flagged ({pair.flagged!r}) must equal "
                    f"holm_adjusted_p < alpha ({pair.holm_adjusted_p!r} < {self.alpha!r} = "
                    f"{expected_flagged!r})"
                )


def check_monotonicity(
    obs: Observations | ContinuousObservations,
    *,
    dependence: str | None = None,
    alpha: float = 0.05,
) -> MonotonicityCheck:
    """Test every pair of adjacent severity levels in ``obs`` for a reversal: a one-sided Fisher
    exact test for "the higher severity performs better" under ``obs.direction``,
    Holm-corrected across the whole family.

    For :class:`~robustmargin.ContinuousObservations` the test between adjacent levels is the
    one-sided Welch t-test on their means (Welch, 1947, *Biometrika* 34:28-35, with the
    Satterthwaite, 1946, degrees of freedom), under the same Holm correction. A pair in which
    either level has a single observation, or whose two levels both have zero spread, has no
    spread to measure and is not testable: its p-value is recorded as ``1.0`` and ``warnings``
    names it. Rounded or quantised responses can produce zero spread, and a test that treated it
    as infinite precision would flag every mean difference.

    Levels are built from ``obs.cells()`` -- successes/trials pooled per exact severity, sorted
    ascending, exactly like every other cell-based computation in this package (aggregated
    and per-observation input give identical results). The zero-severity control is a level like
    any other, and the control-to-first-level pair is always in the family, whether or not the
    likelihood a fit built from the same data used the control row (a fit drops an
    all-success control under a fixed ``upper=1.0``; this function never reads a fit at all).

    Parameters
    ----------
    obs
        The severity/outcome data to test.
    dependence
        ``None`` (the default) resolves to ``"independent"`` unless ``obs.cluster`` is set, in
        which case omitting it raises (a method assuming independence must refuse clustered
        input unless the caller explicitly opts in). ``"independent"`` may always be passed
        explicitly; the resulting ``MonotonicityCheck.dependence`` is always ``"independent"`` in
        this release.
    alpha
        The significance level ``flagged`` is decided at (``holm_adjusted_p < alpha``), strictly
        between 0 and 1. Defaults to ``0.05``.

    Returns
    -------
    MonotonicityCheck
        ``pairs`` is empty (with a warning naming the reason) when ``obs`` has fewer than two
        distinct severity levels.

    Raises
    ------
    ValueError
        If ``dependence`` is not ``None`` or ``"independent"``; if ``obs.cluster`` is set and
        ``dependence`` is ``None``; or if ``alpha`` is not a finite number strictly between 0
        and 1.
    """
    if dependence is not None and dependence not in _VALID_DEPENDENCE_INPUTS:
        raise ValueError(
            "check_monotonicity: dependence must be None or 'independent' in this release, got "
            f"{dependence!r} (the check assumes independent trials; a paired/clustered design "
            "needs its own method, deferred to a future decision)"
        )
    if obs.cluster is not None and dependence is None:
        raise ValueError(
            "check_monotonicity: obs carries cluster ids (obs.cluster is set); a method that "
            "assumes independence must not run on clustered data unless the caller passes "
            "dependence='independent' explicitly"
        )
    _check_finite_number(alpha, label="check_monotonicity alpha")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"check_monotonicity: alpha must satisfy 0 < alpha < 1, got {alpha!r}")

    resolved_dependence = "independent"
    direction = obs.direction
    alternative = "greater" if direction == "decreasing" else "less"
    if isinstance(obs, ContinuousObservations):
        return _check_continuous_monotonicity(
            obs, alpha=alpha, alternative=alternative, dependence=resolved_dependence
        )
    cells = obs.cells()

    if len(cells) < 2:
        return MonotonicityCheck(
            direction=direction,
            alpha=alpha,
            dependence=resolved_dependence,
            pairs=(),
            any_flagged=False,
            warnings=(
                "check_monotonicity: fewer than two distinct severity levels "
                f"({len(cells)}) -- no adjacent pairs to test",
            ),
        )

    level_pairs = list(zip(cells, cells[1:]))  # deliberately not strict: cells[1:] is one shorter
    raw_p_values: list[float] = []
    for low, high in level_pairs:
        failures_low = low.trials - low.successes
        failures_high = high.trials - high.successes
        table = [[high.successes, failures_high], [low.successes, failures_low]]
        _, p_value = fisher_exact(table, alternative=alternative)
        raw_p_values.append(float(p_value))

    _, holm_adjusted, _, _ = multipletests(raw_p_values, method="holm")

    pairs = tuple(
        AdjacentPair(
            severity_low=low.severity,
            severity_high=high.severity,
            successes_low=low.successes,
            trials_low=low.trials,
            successes_high=high.successes,
            trials_high=high.trials,
            p_value=p_value,
            holm_adjusted_p=float(holm_p),
            flagged=bool(holm_p < alpha),
        )
        for (low, high), p_value, holm_p in zip(
            level_pairs, raw_p_values, holm_adjusted, strict=True
        )
    )
    any_flagged = any(pair.flagged for pair in pairs)

    return MonotonicityCheck(
        direction=direction,
        alpha=alpha,
        dependence=resolved_dependence,
        pairs=pairs,
        any_flagged=any_flagged,
        warnings=(),
    )


def _welch_reversal_p_value(
    low: ContinuousCell, high: ContinuousCell, *, alternative: str
) -> tuple[float, float | None, float | None]:
    """``(p_value, statistic, df)`` of the one-sided Welch test for a reversal between two
    adjacent levels. ``statistic`` and ``df`` are ``None`` when no t statistic exists (a level
    with one observation), and the pair is untestable, with p = 1, when both levels have no
    spread."""
    if low.n < 2 or high.n < 2:
        return 1.0, None, None
    variance_low = low.sd**2 / low.n
    variance_high = high.sd**2 / high.n
    total = variance_low + variance_high
    difference = high.mean - low.mean
    if not total > 0.0:
        return 1.0, None, None
    statistic = difference / math.sqrt(total)
    df = 1.0 / (
        (variance_low / total) ** 2 / (low.n - 1) + (variance_high / total) ** 2 / (high.n - 1)
    )
    if alternative == "greater":
        p_value = float(t_distribution.sf(statistic, df))
    else:
        p_value = float(t_distribution.cdf(statistic, df))
    return p_value, float(statistic), float(df)


def _check_continuous_monotonicity(
    obs: ContinuousObservations, *, alpha: float, alternative: str, dependence: str
) -> MonotonicityCheck:
    cells = obs.cells()
    if len(cells) < 2:
        return MonotonicityCheck(
            direction=obs.direction,
            alpha=alpha,
            dependence=dependence,
            pairs=(),
            any_flagged=False,
            method=_CONTINUOUS_MONOTONICITY_METHOD,
            warnings=(
                "check_monotonicity: fewer than two distinct severity levels "
                f"({len(cells)}) -- no adjacent pairs to test",
            ),
        )
    level_pairs = list(zip(cells, cells[1:]))
    tests = [
        _welch_reversal_p_value(low, high, alternative=alternative) for low, high in level_pairs
    ]
    _, holm_adjusted, _, _ = multipletests([p for p, _, _ in tests], method="holm")
    untestable = [
        f"({low.severity!r}, {high.severity!r})"
        for (low, high), (_, statistic, _) in zip(level_pairs, tests)
        if statistic is None
    ]
    notes: tuple[str, ...] = ()
    if untestable:
        notes = (
            "check_monotonicity: severity pair(s) " + ", ".join(untestable) + " have a level "
            "with a single observation or no spread at either level, so no spread can be "
            "measured and the Welch test is not defined; their p-value is recorded as 1.0",
        )
    pairs = tuple(
        ContinuousAdjacentPair(
            severity_low=low.severity,
            severity_high=high.severity,
            n_low=low.n,
            mean_low=low.mean,
            sd_low=low.sd,
            n_high=high.n,
            mean_high=high.mean,
            sd_high=high.sd,
            statistic=statistic,
            df=df,
            p_value=p_value,
            holm_adjusted_p=float(holm_p),
            flagged=bool(holm_p < alpha),
        )
        for (low, high), (p_value, statistic, df), holm_p in zip(
            level_pairs, tests, holm_adjusted, strict=True
        )
    )
    return MonotonicityCheck(
        direction=obs.direction,
        alpha=alpha,
        dependence=dependence,
        pairs=pairs,
        any_flagged=any(pair.flagged for pair in pairs),
        method=_CONTINUOUS_MONOTONICITY_METHOD,
        warnings=notes,
    )


def _success_and_failure_probabilities(fit: Fit) -> tuple[np.ndarray, np.ndarray]:
    """``(p, 1 - p)`` per cell of ``fit.cells``, from the one shared curve formula in
    ``robustmargin.models`` -- never a second copy of it here. Imported
    inside this function, not at module scope, since ``robustmargin.models`` imports this module at
    *its* top level (see this module's own docstring on the import cycle); by the time any
    ``Fit`` exists to diagnose, ``robustmargin.models`` has already finished importing.
    """
    from robustmargin.models import (  # local: see docstring
        _count_cells,
        _curve_success_and_failure_probability,
    )

    assert fit.params is not None  # diagnose() only calls this for an OK fit
    mu = fit.params["mu"].value
    s = fit.params["s"].value
    upper = fit.params["upper"].value
    lower = fit.params["lower"].value
    severities = np.array([cell.severity for cell in _count_cells(fit)], dtype=np.float64)
    return _curve_success_and_failure_probability(
        severities, mu=mu, s=s, upper=upper, lower=lower, scale=fit.axis.scale, link=fit.link
    )


def _refit_link(fit: Fit, link: str, *, mirrored: bool = False) -> Fit:
    """Refit ``fit``'s data under ``link``, holding the same ``upper``/``lower`` specification
    (both fixed/estimate flags and, when fixed, the same value). Goes through the existing
    private statsmodels fitting paths in ``robustmargin.models`` (no new optimizer or IRLS) --
    imported here, inside the function, rather than at module scope, since
    by the time any ``Fit`` exists to diagnose, ``robustmargin.models`` has already finished
    importing this module (see this module's own docstring).

    Skips the separation and control-incompatibility pre-checks ``fit_dose_response`` itself
    runs before calling either fitting path: both are properties of the data and the
    ``upper``/``lower`` configuration, not of the link, and ``fit`` already reached
    :data:`~robustmargin.Status.OK` under the very same data and configuration, so a different link
    cannot newly trigger either one.

    ``mirrored`` says that ``fit`` is the complement of an increasing fit; it changes only the
    wording of the refit's warnings.
    """
    from robustmargin.models import (  # local: see docstring above
        _count_cells,
        _fit_generic,
        _fit_glm,
    )

    assert fit.params is not None  # diagnose() only calls this for an OK fit
    upper_param = fit.params["upper"]
    lower_param = fit.params["lower"]
    upper_is_estimate = not upper_param.fixed
    lower_is_estimate = not lower_param.fixed
    upper_value = math.nan if upper_is_estimate else upper_param.value
    lower_value = math.nan if lower_is_estimate else lower_param.value

    cells = _count_cells(fit)
    obs = Observations.from_counts(
        fit.axis,
        severity=[c.severity for c in cells],
        successes=[c.successes for c in cells],
        trials=[c.trials for c in cells],
        outcome=fit.outcome,
        direction=fit.direction,
    )

    if (
        not upper_is_estimate
        and not lower_is_estimate
        and upper_value == 1.0
        and lower_value == 0.0
    ):
        return _fit_glm(obs, cells=cells, cluster_ids=None, link=link, mirrored=mirrored)
    return _fit_generic(
        obs,
        cells=cells,
        cluster_ids=None,
        link=link,
        upper_is_estimate=upper_is_estimate,
        upper_value=upper_value,
        lower_is_estimate=lower_is_estimate,
        lower_value=lower_value,
        mirrored=mirrored,
    )


def _link_comparison(
    fit: Fit, *, mirrored: bool = False
) -> tuple[dict[str, float | None], dict[str, Status]]:
    """AIC and status for each of ``_COMPARISON_LINKS``: the diagnosed fit's own link reuses its
    own ``log_likelihood``/parameter count directly (exact, no refit); the other two are refit
    via :func:`_refit_link` (so each call does two extra fits).
    """
    assert fit.params is not None and fit.log_likelihood is not None  # fit.status is OK

    link_aic: dict[str, float | None] = {}
    link_status: dict[str, Status] = {}
    for link in _COMPARISON_LINKS:
        if link == fit.link:
            k = sum(1 for p in fit.params.values() if not p.fixed)
            link_aic[link] = -2.0 * fit.log_likelihood + 2.0 * k
            link_status[link] = Status.OK
            continue
        refit = _refit_link(fit, link, mirrored=mirrored)
        if refit.status is Status.OK:
            assert refit.params is not None and refit.log_likelihood is not None
            k = sum(1 for p in refit.params.values() if not p.fixed)
            link_aic[link] = -2.0 * refit.log_likelihood + 2.0 * k
        else:
            link_aic[link] = None
        link_status[link] = refit.status
    return link_aic, link_status


def _is_structural_zero_control(fit: Fit, cell: Cell) -> bool:
    """A cell is excluded from goodness of fit **only** when it is deterministic *by
    construction* under the model, never merely because its *computed* fitted probability
    happens to round to a boundary (a value-based rule would make ``df`` depend on the link).

    The only case that is structurally deterministic in this release: a log-axis cell at severity
    exactly ``0.0``, under ``direction="decreasing"`` (an increasing outcome is diagnosed as its
    complement, which is decreasing), when ``upper`` is *fixed* at exactly ``1.0`` -- the
    ``P(0) = upper`` rule for a zero-severity control then pins that cell's fitted probability to
    exactly ``1.0`` with zero variance, for every possible ``mu``/``s``. An *estimated* ``upper``
    is never
    exempt, even when its fitted value happens to equal ``1.0``: the control still identifies
    ``u`` in that case, so it is not deterministic *by construction* and is counted.
    """
    if cell.severity != 0.0 or fit.axis.scale != "log" or fit.direction != "decreasing":
        return False
    assert fit.params is not None
    upper_param = fit.params["upper"]
    return upper_param.fixed and upper_param.value == 1.0


def _diagnose_gaussian(fit: Fit) -> ContinuousDiagnostics:
    """:func:`diagnose` for a converged Gaussian fit, on its decreasing view (an increasing fit
    is the negation of one; every quantity here is the same for both)."""
    from robustmargin.models import _fit_gaussian_cells, _mirror_fit  # local: see _refit_link

    view = _mirror_fit(fit) if fit.direction == "increasing" else fit
    assert view.params is not None and view.log_likelihood is not None
    assert view.residual is not None
    cells = tuple(c for c in view.cells if isinstance(c, ContinuousCell))
    pure_error_ss = math.fsum(c.sd * c.sd * (c.n - 1) for c in cells)
    rows = sum(c.n for c in cells)
    pure_error_df = rows - len(cells)
    lack_of_fit_df = view.residual.df - pure_error_df
    lack_of_fit_f: float | None = None
    lack_of_fit_p: float | None = None
    if lack_of_fit_df > 0 and pure_error_df > 0 and pure_error_ss > 0.0:
        excess = max(view.residual.rss - pure_error_ss, 0.0)
        lack_of_fit_f = (excess / lack_of_fit_df) / (pure_error_ss / pure_error_df)
        lack_of_fit_p = float(f_distribution.sf(lack_of_fit_f, lack_of_fit_df, pure_error_df))

    upper_param, lower_param = view.params["upper"], view.params["lower"]
    link_aic: dict[str, float | None] = {}
    link_status: dict[str, Status] = {}
    for link in _COMPARISON_LINKS:
        if link == view.link:
            candidate = view
        else:
            candidate = _fit_gaussian_cells(
                cells,
                axis=view.axis,
                outcome=view.outcome,
                direction="decreasing",
                cluster_ids=None,
                link=link,
                upper_is_estimate=not upper_param.fixed,
                upper_value=upper_param.value,
                lower_is_estimate=not lower_param.fixed,
                lower_value=lower_param.value,
            )
        link_status[link] = candidate.status
        if candidate.status is Status.OK:
            assert candidate.params is not None and candidate.log_likelihood is not None
            # The residual variance is one more estimated parameter for every link, so it moves
            # no comparison; it is counted so the value is the usual Gaussian AIC.
            k = sum(1 for p in candidate.params.values() if not p.fixed) + 1
            link_aic[link] = -2.0 * candidate.log_likelihood + 2.0 * k
        else:
            link_aic[link] = None
    return ContinuousDiagnostics(
        rss=view.residual.rss,
        df=view.residual.df,
        pure_error_ss=pure_error_ss,
        pure_error_df=pure_error_df,
        lack_of_fit_f=lack_of_fit_f,
        lack_of_fit_p=lack_of_fit_p,
        link_aic=link_aic,
        link_status=link_status,
    )


def diagnose(fit: Fit) -> Diagnostics | ContinuousDiagnostics:
    """Model-only diagnostics for a converged fit: goodness of fit, dispersion, and link
    comparison. Never changes ``fit`` itself (reported, never acted on) -- in particular,
    ``fit.link`` is never consulted to pick a "better" link.

    A Gaussian fit gets :class:`ContinuousDiagnostics` instead: the residual sum of squares, a
    pure-error lack-of-fit F test (Draper and Smith, *Applied Regression Analysis*, 3rd ed.,
    1998, ch. 2) when the design has replicates, and the link comparison by AIC.

    Parameters
    ----------
    fit
        The fit to diagnose. Must have ``status is Status.OK`` (a failed fit
        carries no model-derived numbers -- :func:`~robustmargin.fit_dose_response` itself only
        calls this for an ``OK`` fit and leaves :attr:`~robustmargin.Fit.diagnostics` ``None``
        otherwise).

    Returns
    -------
    Diagnostics or ContinuousDiagnostics
        :class:`Diagnostics` for a binomial fit and :class:`ContinuousDiagnostics` for a
        Gaussian one.

    Raises
    ------
    ValueError
        If ``fit.status`` is not :data:`~robustmargin.Status.OK`.
    """
    if fit.status is not Status.OK:
        raise ValueError(
            f"diagnose: fit.status must be Status.OK (a failed fit carries no model-derived "
            f"diagnostics), got {fit.status!r}"
        )
    if fit.model == "gaussian":
        return _diagnose_gaussian(fit)
    from robustmargin.models import _count_cells  # local: see _refit_link

    mirrored = fit.direction == "increasing"
    if mirrored:
        # The goodness-of-fit sums, the sparse-cell screen and the link refits all run on the
        # complementary outcome, which is the decreasing model they are written for. Deviance,
        # Pearson chi-square, degrees of freedom and AIC are the same for both views.
        from robustmargin.models import _mirror_fit  # local: see _refit_link

        fit = _mirror_fit(fit)
    assert fit.params is not None and fit.log_likelihood is not None

    k = sum(1 for p in fit.params.values() if not p.fixed)
    p_raw, q_raw = _success_and_failure_probabilities(fit)
    # R's own binomial()$linkinv clamp, applied to each of p and (1 - p) independently -- never
    # derived one from the other's clamped value, which would reintroduce the cancellation
    # _curve_success_and_failure_probability exists to avoid.
    p_clamped = np.clip(p_raw, _PROBABILITY_EPS, 1.0 - _PROBABILITY_EPS)
    q_clamped = np.clip(q_raw, _PROBABILITY_EPS, 1.0 - _PROBABILITY_EPS)

    deviance_sum = 0.0
    pearson_sum = 0.0
    counted_cells = 0
    sparse_entries: list[tuple[float, float]] = []
    for cell, p, q in zip(_count_cells(fit), p_clamped, q_clamped, strict=True):
        if _is_structural_zero_control(fit, cell):
            # Deterministic under the model (P(0) = upper exactly, fixed at 1.0) -- zero
            # variance and zero deviance by construction, not excluded because its *computed*
            # probability happens to round to a boundary.
            continue
        counted_cells += 1
        y = float(cell.successes)
        n = float(cell.trials)
        p_f = float(p)
        q_f = float(q)
        expected_successes = n * p_f
        expected_failures = n * q_f
        deviance_sum += 2.0 * (
            float(xlogy(y, y / expected_successes))
            + float(xlogy(n - y, (n - y) / expected_failures))
        )
        pearson_sum += (y - expected_successes) ** 2 / (n * p_f * q_f)
        # A side is flagged only when *both* its expected count is below the cutoff and
        # something was actually observed there -- a small expectation with nothing observed on
        # that side is a clean plateau, not a finding.
        flagged_side_expecteds: list[float] = []
        if expected_successes < _SPARSE_EXPECTED_CUTOFF and y >= 1.0:
            flagged_side_expecteds.append(expected_successes)
        if expected_failures < _SPARSE_EXPECTED_CUTOFF and (n - y) >= 1.0:
            flagged_side_expecteds.append(expected_failures)
        if flagged_side_expecteds:
            sparse_entries.append((cell.severity, min(flagged_side_expecteds)))

    df = counted_cells - k
    warnings: list[str] = []
    deviance: float | None
    pearson_chi2: float | None
    dispersion: float | None
    if df <= 0:
        deviance, pearson_chi2, dispersion = None, None, None
        warnings.append(
            f"diagnose: degrees of freedom is {df} (<= 0: {counted_cells} cell(s) counted "
            f"minus {k} estimated parameter(s)) -- no goodness-of-fit statistics are reported"
        )
    else:
        deviance = float(deviance_sum)
        pearson_chi2 = float(pearson_sum)
        dispersion = pearson_chi2 / df
        if dispersion > _DISPERSION_WARNING_THRESHOLD:
            warnings.append(
                f"{_OVERDISPERSED_WARNING_PREFIX}dispersion (Pearson chi-square / df) is "
                f"{dispersion!r}, exceeding {_DISPERSION_WARNING_THRESHOLD}; nothing is "
                "rescaled and no interval changes as a result"
            )

    if sparse_entries:
        # Report-only -- names only the flagged severities (an observed event against a
        # below-cutoff expectation on that side), and the single smallest expected count among
        # those *flagged sides only*; changes no number above.
        severities_repr = ", ".join(repr(severity) for severity, _ in sparse_entries)
        minimum_expected = min(count for _, count in sparse_entries)
        warnings.append(
            f"{_SPARSE_CELLS_WARNING_PREFIX}severity/severities {severities_repr} have an "
            f"expected count of successes or failures below "
            f"{_SPARSE_EXPECTED_CUTOFF!r} with at least one observed there (minimum "
            f"{minimum_expected:.6f}); Pearson chi-square and dispersion (phi) are unreliable "
            "there; no number here is changed as a result"
        )

    link_aic, link_status = _link_comparison(fit, mirrored=mirrored)

    return Diagnostics(
        deviance=deviance,
        pearson_chi2=pearson_chi2,
        df=df,
        dispersion=dispersion,
        link_aic=link_aic,
        link_status=link_status,
        warnings=tuple(warnings),
    )
