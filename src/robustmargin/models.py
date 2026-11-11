"""The dose-response fit result, its fitting entry point, and prediction.

This module holds :class:`Fit`, the two small dataclasses it embeds (:class:`Parameter`,
:class:`Covariance`), :func:`fit_dose_response`, and :class:`Prediction` (:meth:`Fit.predict`'s
return type). Two fitting paths exist: statsmodels ``GLM`` when both asymptotes are fixed at
exactly ``1.0``/``0.0``, and ``statsmodels.base.model.GenericLikelihoodModel`` -- given a
likelihood function, never a hand-rolled optimizer or IRLS loop -- otherwise. A fit with no
interior maximum (wrong-sign slope, or an estimated asymptote pinned to a parameter-space
boundary) is reported as ``Status.NOT_CONVERGED`` with the reason in ``Fit.warnings``, never as
a number that happens to be where the optimizer stopped.

A curve that is a step between adjacent tested levels, with an estimated asymptote or a fixed
one other than ``1``/``0``, has no finite maximum: the log-likelihood keeps rising as ``s -> 0``.
Such a fit is reported as ``Status.NOT_CONVERGED``, found by comparing the fitted
log-likelihood with the closed-form supremum on the ``s -> 0`` boundary. The pre-fit overlap check
is exact only for ``u = 1, l = 0``.

**Two model families.** ``model="binomial"`` is fitted to counts (:class:`Observations`) and
``model="gaussian"`` to real-valued responses (:class:`ContinuousObservations`). The Gaussian
model is the same curve, ``P(x) = u - (u - l) * F(z)``, with independent homoscedastic normal
errors, fitted by nonlinear least squares; ``logit``, ``probit`` and ``cloglog`` are the
parametrisations R ``drc`` calls ``LL.4``, ``LN.4`` and ``W1.4``. Its covariance is the residual
variance times the inverse observed Hessian of the residual sum of squares (cf. Ritz, Baty,
Streibig and Gerhard, 2015, *PLOS ONE* 10(12):e0146021), calibrated on the residual degrees of
freedom. A Gaussian fit that cannot be identified is ``Status.NOT_CONVERGED`` with a named reason
and no number; there is no separation, no exact bound and no fallback to another model.
:data:`robustmargin.report.SCHEMA_VERSION` is the version :func:`robustmargin.report.to_dict`
writes and the single source of that number.

**Outcomes that rise with severity.** For ``Observations(direction="increasing")`` the counted
outcome rises with severity and the curve is ``P(x) = l + (u - l) * F(z)``. It is fit as the
decreasing curve of the complementary outcome (``trials - successes``, asymptotes ``1 - l`` and
``1 - u``) and mapped back, so the likelihood, the optimisers and every status rule are the ones the
decreasing case uses; no separate estimator exists. The mapping is kept in :func:`_mirror_fit`.

:class:`Fit` also carries :attr:`~Fit.diagnostics` (model-only goodness of fit, dispersion and
link comparison, filled only when ``status`` is ``OK`` -- see :func:`robustmargin.diagnose`) and
:attr:`~Fit.monotonicity` (a data-only adjacent-level check, filled on *every* return path
regardless of ``status`` -- see :func:`robustmargin.check_monotonicity` in
:mod:`robustmargin.validation`). Both are reported and never acted on: neither changes ``link``,
``params``, or any other field of the ``Fit`` they are attached to.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, NamedTuple, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import expit, gammaln, log_expit, xlog1py, xlogy
from scipy.special import logit as _logit
from scipy.stats import norm
from scipy.stats import t as student_t
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning
from statsmodels.tools.tools import add_constant

from robustmargin.types import (
    Axis,
    Cell,
    ContinuousCell,
    ContinuousObservations,
    JSONValue,
    Observations,
    Residual,
    Status,
    _check_finite_number,
)
from robustmargin.validation import (
    ContinuousDiagnostics,
    Diagnostics,
    MonotonicityCheck,
    check_monotonicity,
    diagnose,
)

# `robustmargin.validation` only imports `Fit` from this module under `TYPE_CHECKING` (see that
# module's own docstring), so this is not a circular import at runtime: by the time
# `fit_dose_response` below actually calls `diagnose`/`check_monotonicity`, this module has
# already finished executing this very import statement.

# Narrow statsmodels imports above, never `statsmodels.api`: that module also imports graphics and
# tsa, and statsmodels 0.14.0's tsa fails to import against current pandas
# (`deprecate_kwarg() missing 'new_arg_name'`), which breaks the dependency-floor CI job.

__all__ = ["Covariance", "Fit", "Parameter", "Prediction", "fit_dose_response"]

_VALID_MODELS = frozenset({"binomial", "gaussian"})
_VALID_LINKS: dict[str, frozenset[str]] = {
    "binomial": frozenset({"probit", "logit", "cloglog"}),
    "gaussian": frozenset({"probit", "logit", "cloglog"}),
}
# A Gaussian fit has no separation, no exact-bound fallback and no control incompatibility: it
# either identifies the curve or says, with a named reason, that it cannot.
_GAUSSIAN_FIT_STATUSES = frozenset({Status.OK, Status.NOT_CONVERGED})
_BINOMIAL_PARAM_NAMES = frozenset({"mu", "s", "upper", "lower"})
_VALID_DIRECTIONS = frozenset({"decreasing", "increasing"})
_INVALID_FIT_STATUSES = frozenset({Status.UNREACHABLE, Status.FAILS_AT_BASELINE})

# Tolerance for Covariance's positive-definiteness check: the minimum eigenvalue of the
# correlation matrix ``D^-1/2 . matrix . D^-1/2`` must exceed this, unscaled, so the test does
# not depend on the severity units. Strictly positive (not merely semi-definite): a singular
# matrix, such as ``((0.0, 0.0), (0.0, 1.0))``, is rejected, because a singular
# covariance means at least one estimated parameter carries no information at all.
_POSITIVE_DEFINITE_RELATIVE_TOLERANCE = 1e-12

# ------------------------------------------------------------------------------------------
# fit_dose_response: module constants (the numeric boundary test for an estimated asymptote)
# ------------------------------------------------------------------------------------------

_VALID_UPPER_LOWER = "estimate"
_STATSMODELS_LINKS = {
    "probit": families.links.Probit(),
    "logit": families.links.Logit(),
    "cloglog": families.links.CLogLog(),
}
# An estimated asymptote's logit-scale parameter (a for lower, b for upper) beyond this
# magnitude, or the asymptote itself within this distance of a probability boundary (0 or 1),
# means the likelihood has no interior maximum -- reported as Status.NOT_CONVERGED with a
# boundary warning, never as a number. Changing either cut-off is a statistical-method change.
_ASYMPTOTE_LOGIT_BOUND = 15.0
_ASYMPTOTE_PROBABILITY_EPS = 1e-6
# Separation is decided by the exact pre-fit overlap check (`_check_separation`), never by
# statsmodels' PerfectSeparationWarning. That warning tests `allclose(fitted, observed)`, which on
# grouped binomial data also fires on an exact fit -- every two-level design with overlap -- so it
# is filtered to "ignore" around both GLM calls below. Every fit reaching those calls has already
# passed the exact check. The check is exact for fixed u = 1, l = 0 (Albert & Anderson, 1984;
# Silvapulle, 1981, for the existence of a finite maximum under log-concave links); with
# estimated or other fixed asymptotes it is necessary but not sufficient, and a curve that is a
# step between adjacent levels there is detected after the fit (see the module docstring).

# A fitted `s` greater than this factor times the span of the likelihood-contributing transformed
# levels (the tested range on `log x` or `x`, excluding a log-axis zero-severity control, which
# carries no covariate value) is flat up to rounding -- the wrong-sign case's degenerate limit,
# `s -> infinity` -- and gives Status.NOT_CONVERGED rather than a covariance built from a near-zero
# slope (which would otherwise pass Covariance's own positive-definiteness check while being
# numerically meaningless, for example ~1e61). Changing this cut-off is a statistical-method change.
_FLAT_SLOPE_SPAN_FACTOR = 1e8

# Likelihood-path convergence: Newton's own `converged` flag only bounds the last step taken, so
# two more checks apply at the final parameters. The Newton polish's log-likelihood must not
# have regressed below the BFGS stage's, within this *relative* tolerance (an absolute tolerance
# does not scale with the log-likelihood's own magnitude, which grows with sample size and is
# not something a caller controls), applied as
# `bfgs_llf - _LLF_REGRESSION_TOLERANCE * (1 + |bfgs_llf|)` -- floating-point noise between two
# different optimizers evaluating the same likelihood at very close points, scaled the same way
# `_SCORE_TOLERANCE_FACTOR` below is.
_LLF_REGRESSION_TOLERANCE = 1e-8
# The score (gradient of the log-likelihood) at the final parameters must be near zero. There is
# no universal absolute tolerance for a gradient, because its scale depends on the
# log-likelihood's own scale (more data means both the log-likelihood and its gradient are
# larger in magnitude even at the true optimum); `1 + |llf|` is a defensible, simple proxy for
# that scale that never reaches zero, so the tolerance below is always well-defined.
_SCORE_TOLERANCE_FACTOR = 1e-4


# Shared wording for a wrong-sign slope (the GLM path's own beta1 <= 0 check) and for the flat
# curve it degenerates into (`_FLAT_SLOPE_SPAN_FACTOR` above) -- the same underlying phenomenon
# (no finite (mu, s) exists once the constrained maximum is pushed to the s -> infinity
# boundary), so one reason is used for both, on both fitting paths.
def _wrong_sign_or_flat_warning(*, mirrored: bool) -> str:
    direction = "increasing" if mirrored else "decreasing"
    return (
        f"fit_dose_response: the fitted curve has no interior maximum for direction={direction!r} "
        "(a wrong-sign slope, or a curve that is flat up to rounding) -- no finite (mu, s) exists"
    )


@dataclass(frozen=True, kw_only=True)
class Parameter:
    """One fitted or fixed parameter value, on the **natural scale** -- the curve's own units
    (for example ``s`` in log-severity units, ``upper``/``lower`` as probabilities), never the
    optimiser's internal reparametrisation (``log s``, or ``upper``/``lower`` on a logit scale).
    An estimated-asymptote fit delta-transforms its optimiser-scale covariance onto this scale
    before it is ever stored here (see :class:`Fit`).

    Attributes
    ----------
    value
        The parameter's value on its natural scale. Must be finite, and must be a genuine
        ``float`` or ``int``, never a ``bool``.
    fixed
        ``True`` if this parameter was held fixed rather than estimated (for example
        ``upper=1.0`` fixed in a two-parameter probit). Must be a genuine ``bool``.
    """

    value: float
    fixed: bool

    def __post_init__(self) -> None:
        _check_finite_number(self.value, label="Parameter.value")
        if not isinstance(self.fixed, bool):
            raise ValueError(f"Parameter.fixed must be a real bool, got {self.fixed!r}")


@dataclass(frozen=True, kw_only=True)
class Covariance:
    """The variance-covariance matrix of a fit's estimated (non-fixed) parameters, on the same
    natural scale as :class:`Parameter`.

    Attributes
    ----------
    names
        The estimated parameter names, in the order ``matrix`` is indexed by. Unique.
    matrix
        A square, finite, symmetric, **positive-definite** matrix with side ``len(names)``.
        Every entry must be a genuine finite number, never a ``bool``. Symmetry is checked with
        :func:`math.isclose` (``rel_tol=1e-9``, ``abs_tol=1e-12``), not exact equality, to
        tolerate floating-point round-trip noise. Positive-definiteness is checked on the
        **correlation matrix** ``D^-1/2 . matrix . D^-1/2`` (``D = diag(matrix)``): every
        variance must be positive, and the correlation matrix's minimum eigenvalue (via
        :func:`numpy.linalg.eigvalsh`) must exceed ``1e-12``. The rule is therefore invariant to
        rescaling any single parameter, which matters because the natural scale mixes severity
        units (``mu``, ``s``) with probabilities (``upper``, ``lower``). It accepts every matrix
        that a raw-eigenvalue rule (minimum eigenvalue ``> 1e-12 * max(1, max|entry|)``) would
        accept. A singular matrix -- for example ``((0.0, 0.0), (0.0, 1.0))``, or any matrix with a
        zero eigenvalue -- is rejected, not merely a matrix with a negative one.
    """

    names: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "matrix", tuple(tuple(row) for row in self.matrix))

        if len(set(self.names)) != len(self.names):
            raise ValueError(f"Covariance.names must be unique, got {self.names!r}")

        n = len(self.names)
        if len(self.matrix) != n:
            raise ValueError(
                f"Covariance.matrix must have {n} rows to match names, got {len(self.matrix)}"
            )
        for row in self.matrix:
            if len(row) != n:
                raise ValueError(
                    f"Covariance.matrix must be square with side {n}, got a row of length "
                    f"{len(row)}"
                )
            for entry in row:
                _check_finite_number(entry, label="Covariance.matrix entries")

        for i in range(n):
            for j in range(n):
                if not math.isclose(
                    self.matrix[i][j], self.matrix[j][i], rel_tol=1e-9, abs_tol=1e-12
                ):
                    raise ValueError(
                        "Covariance.matrix must be symmetric: "
                        f"matrix[{i}][{j}]={self.matrix[i][j]!r} != "
                        f"matrix[{j}][{i}]={self.matrix[j][i]!r}"
                    )

        if n > 0 and not _is_positive_definite(np.array(self.matrix, dtype=float)):
            raise ValueError(
                "Covariance.matrix must be positive-definite: every variance must be positive "
                "and the correlation matrix's minimum eigenvalue must exceed "
                f"{_POSITIVE_DEFINITE_RELATIVE_TOLERANCE!r}"
            )


def _check_cluster_ids(cluster_ids: tuple[str | int, ...] | None, *, label: str) -> None:
    if cluster_ids is None:
        return
    if len(cluster_ids) == 0:
        raise ValueError(f"{label} must be non-empty when given, got an empty tuple")
    if len(set(cluster_ids)) != len(cluster_ids):
        raise ValueError(f"{label} must be unique, got {cluster_ids!r}")
    for cluster_id in cluster_ids:
        if cluster_id is None or isinstance(cluster_id, bool):
            raise ValueError(
                f"{label} elements must be a str or int, never None or bool, got {cluster_id!r}"
            )
        if not isinstance(cluster_id, (str, int)):
            raise ValueError(f"{label} elements must be a str or int, got {cluster_id!r}")


@dataclass(frozen=True, kw_only=True, eq=False)
class Prediction:
    """The curve and a confidence band evaluated at a set of severities (:meth:`Fit.predict`).
    Never serialised (``report.py`` does not know this type): it is a throwaway array bundle,
    not a stored result.

    Two band scales are used, chosen by :meth:`Fit.predict` from which asymptotes were
    estimated -- documented in full on :meth:`Fit.predict` itself, since the choice depends on
    the ``Fit`` being predicted from, not on this type.

    ``eq``/``hash`` are disabled (unlike every other result dataclass in this package): the
    default dataclass equality would compare ``numpy`` arrays element-by-element and then try to
    coerce that into a single ``bool``, which raises. Two ``Prediction`` instances are therefore
    never equal unless they are the same object (Python's default identity-based ``__eq__``,
    from ``object``, applies instead) -- do not write ``fit.predict(x) == fit.predict(x)``
    expecting a value comparison; compare the four arrays individually instead.

    All four arrays are read-only (``.flags.writeable is False``): this is a frozen dataclass,
    but freezing only stops an attribute from being *reassigned* (``prediction.lo = ...``), not
    the ``numpy`` array it refers to from being mutated in place (``prediction.lo[:] = ...``),
    so ``__post_init__`` marks each array non-writeable itself to close that gap.

    A scalar ``severity`` argument to :meth:`Fit.predict` (for example ``fit.predict(0.5)``,
    rather than a list or array) gives 0-d arrays here (``severity.shape == ()``), following
    :func:`numpy.asarray`'s own behaviour for a scalar input; it is never promoted to shape
    ``(1,)``.

    Attributes
    ----------
    severity
        The severities this prediction was evaluated at, as given (as ``float64``).
    estimate
        The fitted curve's value at each ``severity``, same shape as ``severity``.
    lo
        The band's lower side at each ``severity``. ``lo <= estimate`` everywhere.
    hi
        The band's upper side at each ``severity``. ``estimate <= hi`` everywhere.
    level
        The two-sided confidence level the band was built at, strictly between 0 and 1.
    """

    severity: NDArray[np.float64]
    estimate: NDArray[np.float64]
    lo: NDArray[np.float64]
    hi: NDArray[np.float64]
    level: float

    def __post_init__(self) -> None:
        for array in (self.severity, self.estimate, self.lo, self.hi):
            array.flags.writeable = False


@dataclass(frozen=True, kw_only=True)
class Fit:
    """A dose-response fit, or an explicit record of why one was not obtained.

    For ``direction="decreasing"`` the fitted curve is ``P(x) = u - (u - l) * F(z)``, and for
    ``direction="increasing"`` it is ``P(x) = l + (u - l) * F(z)``, where
    ``z = (log(x) - mu) / s`` on a log axis or ``z = (x - mu) / s`` on a linear axis, ``s > 0``,
    and ``F`` is the link's CDF
    (the standard normal CDF for ``probit``, the logistic CDF for ``logit``, or
    ``1 - exp(-exp(z))`` for ``cloglog``, which models the probability of *failure* rising with
    severity (of the counted outcome itself for an increasing curve) and so is not symmetric the
    way ``probit``/``logit`` are). ``mu`` is on the
    transformed axis (``log x`` on a log axis, ``x`` on a linear axis), not on the original
    severity scale. At ``z = 0`` -- severity ``exp(mu)`` on a log axis, or severity ``mu`` on a
    linear axis -- ``probit`` and ``logit`` both give ``P = (u + l) / 2``: the midpoint of the
    fitted dynamic range, :meth:`~robustmargin.Definition.relative` ``(0.5)``, for **any**
    asymptotes ``u``, ``l``, because the standard normal and logistic CDFs each equal ``0.5``
    at ``z = 0``. That midpoint equals :meth:`~robustmargin.Definition.absolute` ``(0.5)`` and
    :meth:`~robustmargin.Definition.baseline_fraction` ``(0.5)`` only when ``u = 1`` and
    ``l = 0``. Under ``cloglog``, which is not symmetric around ``z = 0``,
    ``F(0) = 1 - exp(-1)`` (about ``0.632``), so ``mu`` is never the midpoint of the dynamic
    range, for any asymptotes. Each entry in ``params`` records whether it was held fixed or
    estimated, and is on the **natural scale**, never the optimiser's internal
    reparametrisation (``log s``; ``upper``/``lower`` on a logit scale) -- see
    :class:`Parameter`. For an increasing outcome ``lower`` is the rate at zero severity and
    ``upper`` the rate the curve approaches as severity grows, so ``P(0) = l`` on a log axis.
    ``covariance`` is on that same natural scale: when asymptotes are estimated, the
    optimiser's own covariance (which is on its reparametrised scale) is
    delta-transformed onto the natural scale before being stored here.

    ``status == Status.OK`` if and only if both ``params`` and ``log_likelihood`` are present;
    when ``status`` is anything else, ``params``, ``covariance`` and ``log_likelihood`` are all
    ``None`` -- no number is ever produced for a failed fit. ``Status.UNREACHABLE`` and
    ``Status.FAILS_AT_BASELINE`` describe a *threshold*, not a fit (they depend on a ``Definition``
    a ``Fit`` does not have), and are never a ``Fit.status``.

    The two directions are one model seen from either side: an increasing fit has the same
    ``mu``, ``s``, link, log-likelihood and status as the decreasing fit of the complementary
    outcome (``trials - successes``) with asymptotes ``1 - l`` and ``1 - u``. ``params`` and
    ``covariance`` are reported for the outcome as counted, and ``cells`` holds the counts as
    given.

    ``diagnostics`` (goodness of fit, dispersion, link comparison) and ``monotonicity`` (the
    adjacent-level monotonicity check) are two fields that both default to ``None``. The schema's
    ``Fit`` object has ``additionalProperties: false``, so a card whose ``Fit`` carries a key the
    schema does not define is rejected, including a key with a ``null`` value. Adding a field to
    ``Fit`` is therefore a ``schema_version`` change.

    Attributes
    ----------
    axis
        The :class:`~robustmargin.Axis` the fit was made on.
    outcome
        The outcome name this fit was made against, carried over from the
        :class:`~robustmargin.Observations` it was fit to.
    direction
        ``"decreasing"`` (performance falls as severity rises) or ``"increasing"`` (the counted
        outcome rises with severity).
    model
        The model family: ``"binomial"`` (counts) or ``"gaussian"`` (real-valued responses). A
        Gaussian fit's ``upper`` and ``lower`` are any finite response values with ``lower <
        upper``, its ``cells`` are :class:`~robustmargin.ContinuousCell`, its ``diagnostics`` are
        :class:`~robustmargin.validation.ContinuousDiagnostics`, and its ``residual`` is set.
        A Gaussian fit's ``status`` is ``OK`` or ``NOT_CONVERGED`` only.
    link
        The link function: ``"probit"``, ``"logit"`` or ``"cloglog"``.
        Required; the schema's ``link`` enum carries no ``null`` alternative, so this is a
        genuine ``str``, never ``None``.
    params
        A mapping with exactly the keys ``{"mu", "s", "upper", "lower"}`` when ``status`` is
        ``OK``, else ``None``. ``s.value`` must be positive; for ``model="binomial"``
        ``lower.value`` and ``upper.value`` must each lie in ``[0, 1]`` with ``lower.value <
        upper.value``, and for ``model="gaussian"`` they are finite with ``lower.value <
        upper.value``.
        ``params["mu"].fixed`` and ``params["s"].fixed`` must both be ``False``: this package
        never produces (and this dataclass never accepts) an ``OK`` fit with a fixed location
        or scale.
    covariance
        The estimated parameters' covariance. Required (not ``None``) when ``status`` is
        ``OK``; ``None`` otherwise. If present, ``covariance.names`` is exactly the set of
        non-fixed parameter names.
    log_likelihood
        The fitted log-likelihood, or ``None`` when ``status`` is not ``OK``. Finite when
        present.
    cells
        The per-severity-level pooled data the fit was made from
        (:meth:`~robustmargin.Observations.cells`, or
        :meth:`~robustmargin.ContinuousObservations.cells` for a Gaussian fit). Never empty, and
        in strictly increasing
        order of severity (one cell per tested level), which is checked on construction and
        never repaired by sorting or pooling.
    status
        The :class:`~robustmargin.Status` of this fit. Never ``UNREACHABLE`` or
        ``FAILS_AT_BASELINE`` (those describe a threshold, not a fit).
    cluster_ids
        Optional cluster ids the fitted data carries, one instance per distinct cluster (not
        one per row): unique, non-empty when given, and never ``None`` or ``bool`` elements.
        Set whenever the input :class:`~robustmargin.Observations` carried cluster ids, whatever
        dependence was assumed when this fit was produced -- this field reflects the data's
        structure, not a modelling choice. ``None`` means only that the data had no clusters,
        not that independence was assumed. Used by :class:`~robustmargin.Ratio`'s dependence
        guard.
    warnings
        Free-text notes surfaced alongside the fit (for example, ``CONTROL_INCOMPATIBLE``
        carries a message suggesting ``upper="estimate"``, or ``lower="estimate"`` for an
        increasing outcome). Empty by default.
    schema_version
        The serialised-result schema version this object belongs to. Defaults to ``"2"``, the
        version ``to_dict`` writes; ``from_dict`` also reads ``"1"``.
    provenance
        An opaque mapping the caller may attach to record where the inputs came from.
        robustmargin stores it and never interprets it. Empty by default.
    diagnostics
        Model-only goodness-of-fit, dispersion and link-comparison diagnostics
        (:func:`robustmargin.diagnose`). Set exactly when ``status`` is
        :data:`~robustmargin.Status.OK` (no model-derived numbers on a failed fit); ``None``
        otherwise.
    monotonicity
        A data-only adjacent-severity-level monotonicity check
        (:func:`robustmargin.check_monotonicity`), computed from the raw
        :class:`~robustmargin.Observations` before a fitting path was even chosen. Unlike
        ``diagnostics``, this is set on *every* fresh result regardless of ``status`` -- it needs
        only counts, never a converged fit. A serialised card may omit it (the schema does not
        require it), in which case it reads back as ``None``.
    residual
        A Gaussian fit's residual variance (:class:`~robustmargin.Residual`: ``sigma``, ``df``,
        ``rss``), set exactly when ``model="gaussian"`` and ``status`` is ``OK``; ``None`` for a
        binomial fit and for a failed Gaussian fit. It is what a Gaussian fit's t and F
        calibration uses. The last field.
    """

    axis: Axis
    outcome: str
    direction: str
    model: str
    link: str
    params: Mapping[str, Parameter] | None
    covariance: Covariance | None
    log_likelihood: float | None
    cells: tuple[Cell | ContinuousCell, ...]
    status: Status
    cluster_ids: tuple[str | int, ...] | None = None
    warnings: tuple[str, ...] = ()
    schema_version: str = "2"
    provenance: Mapping[str, JSONValue] = field(default_factory=dict)
    diagnostics: Diagnostics | ContinuousDiagnostics | None = None
    monotonicity: MonotonicityCheck | None = None
    residual: Residual | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "cells", tuple(self.cells))
        if self.cluster_ids is not None:
            object.__setattr__(self, "cluster_ids", tuple(self.cluster_ids))

        if not self.outcome:
            raise ValueError("Fit.outcome must be non-empty")
        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"Fit.direction must be one of {sorted(_VALID_DIRECTIONS)}, got {self.direction!r}"
            )

        if self.model not in _VALID_MODELS:
            raise ValueError(
                f"Fit.model must be one of {sorted(_VALID_MODELS)}, got {self.model!r}"
            )
        allowed_links = _VALID_LINKS[self.model]
        if self.link not in allowed_links:
            raise ValueError(
                f"Fit.link for model={self.model!r} must be one of {sorted(allowed_links)}, "
                f"got {self.link!r}"
            )

        if self.status in _INVALID_FIT_STATUSES:
            raise ValueError(
                f"Fit.status must not be {self.status!r}: that status describes a threshold, "
                "not a fit"
            )
        gaussian = self.model == "gaussian"
        if gaussian and self.status not in _GAUSSIAN_FIT_STATUSES:
            raise ValueError(
                f"Fit.status for model='gaussian' must be one of "
                f"{sorted(s.value for s in _GAUSSIAN_FIT_STATUSES)}, got {self.status!r}: a "
                "Gaussian fit has no separation, no control incompatibility and no exact bound"
            )

        if not self.cells:
            raise ValueError("Fit.cells must not be empty")
        cell_type = ContinuousCell if gaussian else Cell
        for cell in self.cells:
            if not isinstance(cell, cell_type):
                raise ValueError(
                    f"Fit.cells for model={self.model!r} must each be a {cell_type.__name__}, "
                    f"got {type(cell).__name__}"
                )
        for index in range(1, len(self.cells)):
            previous = self.cells[index - 1].severity
            current = self.cells[index].severity
            if not current > previous:
                raise ValueError(
                    "Fit.cells must be in strictly increasing order of severity (one cell per "
                    f"tested level), but cells[{index}].severity={current!r} follows "
                    f"cells[{index - 1}].severity={previous!r}"
                )

        _check_cluster_ids(self.cluster_ids, label="Fit.cluster_ids")

        if self.status is not Status.OK and self.diagnostics is not None:
            raise ValueError(
                "Fit.diagnostics must be None when status is not OK (no model-derived "
                f"numbers on a failed fit), got status={self.status!r} with diagnostics set"
            )
        expected_diagnostics = ContinuousDiagnostics if gaussian else Diagnostics
        if self.diagnostics is not None and not isinstance(self.diagnostics, expected_diagnostics):
            raise ValueError(
                f"Fit.diagnostics for model={self.model!r} must be a "
                f"{expected_diagnostics.__name__}, got {type(self.diagnostics).__name__}"
            )
        if (self.residual is not None) != (gaussian and self.status is Status.OK):
            raise ValueError(
                "Fit.residual is set exactly when model is 'gaussian' and status is OK: got "
                f"model={self.model!r}, status={self.status!r}, residual is not None="
                f"{self.residual is not None!r}"
            )

        is_ok = self.status is Status.OK
        has_params = self.params is not None
        has_log_likelihood = self.log_likelihood is not None
        if is_ok != (has_params and has_log_likelihood):
            raise ValueError(
                "Fit.status is OK if and only if both params and log_likelihood are present: "
                f"status={self.status!r}, params is not None={has_params!r}, "
                f"log_likelihood is not None={has_log_likelihood!r}"
            )
        if not is_ok:
            if self.params is not None or self.covariance is not None:
                raise ValueError(
                    "Fit.params and Fit.covariance must both be None when status is not OK"
                )
            return

        if self.log_likelihood is not None:
            _check_finite_number(self.log_likelihood, label="Fit.log_likelihood")

        if self.covariance is None:
            raise ValueError("Fit.covariance is required when status is OK")

        params = self.params
        assert params is not None  # established by is_ok above
        if set(params) != _BINOMIAL_PARAM_NAMES:
            raise ValueError(
                f"Fit.params for model={self.model!r} must have exactly the keys "
                f"{sorted(_BINOMIAL_PARAM_NAMES)}, got {sorted(params)}"
            )
        if params["mu"].fixed or params["s"].fixed:
            raise ValueError(
                "Fit.params['mu'].fixed and Fit.params['s'].fixed must both be False when "
                "status is OK: this package never produces a fit with a fixed location or "
                "scale"
            )
        s = params["s"].value
        lower = params["lower"].value
        upper = params["upper"].value
        if not s > 0.0:
            raise ValueError(f"Fit.params['s'].value must be positive, got {s!r}")
        if gaussian:
            if not lower < upper:
                raise ValueError(
                    "Fit.params must satisfy lower.value < upper.value for model='gaussian', got "
                    f"lower={lower!r}, upper={upper!r}"
                )
        else:
            if not (0.0 <= lower < upper <= 1.0):
                raise ValueError(
                    "Fit.params must satisfy 0 <= lower.value < upper.value <= 1, got "
                    f"lower={lower!r}, upper={upper!r}"
                )

            if self.axis.scale == "log" and self.direction == "decreasing":
                if upper == 1.0 and params["upper"].fixed:
                    for cell in self.cells:
                        assert isinstance(cell, Cell)
                        if cell.severity == 0.0 and cell.successes < cell.trials:
                            raise ValueError(
                                "Fit: the zero-severity control has failures but 'upper' is "
                                "fixed at exactly 1.0, so the likelihood is zero for this fit on "
                                'a log axis; refit with upper="estimate"'
                            )
            elif self.axis.scale == "log" and lower == 0.0 and params["lower"].fixed:
                for cell in self.cells:
                    assert isinstance(cell, Cell)
                    if cell.severity == 0.0 and cell.successes > 0:
                        raise ValueError(
                            "Fit: the zero-severity control has events but 'lower' is fixed at "
                            "exactly 0.0, so the likelihood is zero for this fit on a log axis; "
                            'refit with lower="estimate"'
                        )

        non_fixed = {name for name, param in params.items() if not param.fixed}
        if set(self.covariance.names) != non_fixed:
            raise ValueError(
                "Fit.covariance.names must equal the non-fixed parameter names: "
                f"expected {sorted(non_fixed)}, got {sorted(self.covariance.names)}"
            )

    def predict(self, severity: ArrayLike, *, level: float = 0.95) -> Prediction:
        """Evaluate the fitted curve and a delta-method confidence band at each of
        ``severity``.

        Two band scales are used, depending on which asymptotes this fit estimated:

        - **Both asymptotes fixed:** the delta method is applied to the linear predictor ``z``
          (using ``mu``'s and ``s``'s covariance), then mapped through
          ``P = u - (u - l) * F(z)``. Because ``u`` and ``l`` are constants here, the band is
          guaranteed to stay inside ``(l, u)`` -- not just ``(0, 1)`` -- for every severity.
        - **Either asymptote estimated:** the delta method is applied to ``logit(P)``, using
          the gradient of ``logit(P)`` with respect to every *estimated* natural parameter
          (``mu``, ``s``, and whichever of ``lower``/``upper`` are not fixed), then
          back-transformed with the logistic function. The band is guaranteed to stay inside
          ``(0, 1)``, but not inside ``(l, u)`` specifically, since ``l``/``u`` themselves now
          carry uncertainty.

        For an increasing outcome the curve and its band are those of the complementary outcome
        mapped back (``estimate = 1 - estimate'``, ``lo = 1 - hi'``, ``hi = 1 - lo'``), with the
        same two band scales.

        **A Gaussian fit** has one band scale, the response scale: ``estimate +- t * se``, with
        ``se`` the delta-method standard error from the gradient of the curve in the estimated
        natural parameters and ``t`` the Student-t quantile on the residual degrees of freedom.
        The band is not clipped to any interval. An increasing Gaussian fit is the negation of
        the decreasing fit of the negated response.

        On a log axis, severity ``0.0`` is always the zero-severity control: its estimate
        is ``upper`` exactly (``lower`` for an increasing outcome), never computed through
        ``log(0)``. It is not a special case for
        the band, either: it goes through the same computation as every other severity (so
        ``predict(0)`` and ``predict(1e-12)`` agree), which happens to give a zero-width band
        there whenever ``upper`` is fixed (perturbing any other parameter cannot move a
        constant) and a band from ``Var(upper)`` -- correctly propagated through ``logit``, not
        treated as if it were already on the logit scale -- when ``upper`` is estimated.

        Parameters
        ----------
        severity
            Severities to evaluate the curve at. Converted to ``float64``; every entry must be
            finite and non-negative.
        level
            The two-sided confidence level for the band, strictly between 0 and 1 (not a
            ``bool``). Defaults to ``0.95``.

        Returns
        -------
        Prediction
            ``severity``, ``estimate``, ``lo`` and ``hi`` are ``numpy`` arrays of the same
            shape as the (converted) ``severity`` input; ``lo <= estimate <= hi`` everywhere.

        Raises
        ------
        ValueError
            If ``self.status`` is not :data:`~robustmargin.Status.OK` (a failed fit has no curve
            to evaluate), if ``level`` is not finite, is a ``bool``, or
            does not satisfy ``0 < level < 1``, or if ``severity`` is empty or contains a
            negative or non-finite entry.
        """
        if self.status is not Status.OK:
            raise ValueError(
                f"Fit.predict requires status=OK, got {self.status!r}: a failed fit has no "
                "curve to evaluate"
            )
        _check_finite_number(level, label="Fit.predict level")
        if not (0.0 < level < 1.0):
            raise ValueError(f"Fit.predict: level must satisfy 0 < level < 1, got {level!r}")

        # A copy (`np.array`, not `np.asarray`): Prediction.severity is marked read-only below,
        # and a caller who happened to pass an existing float64 array must never have that
        # array silently frozen out from under them.
        x = np.array(severity, dtype=np.float64)
        if x.size == 0:
            raise ValueError("Fit.predict: severity must not be empty")
        if not bool(np.all(np.isfinite(x))):
            raise ValueError("Fit.predict: severity must be finite (no NaN or inf)")
        if bool(np.any(x < 0.0)):
            raise ValueError("Fit.predict: severity must be non-negative")

        if self.model == "gaussian":
            return self._predict_gaussian(x, level=level)

        if self.direction == "increasing":
            mirrored_band = _mirror_fit(self).predict(x, level=level)
            estimate = 1.0 - mirrored_band.estimate
            lo = 1.0 - mirrored_band.hi
            hi = 1.0 - mirrored_band.lo
            if self.axis.scale == "log":
                # The control is exactly `lower`, not `1 - (1 - lower)`. A fixed `lower` has no
                # uncertainty there, so its band has zero width; an estimated one keeps its band,
                # widened if rounding left the estimate outside it.
                assert self.params is not None
                lower_param = self.params["lower"]
                at_control = x == 0.0
                estimate = np.where(at_control, lower_param.value, estimate)
                if lower_param.fixed:
                    lo = np.where(at_control, lower_param.value, lo)
                    hi = np.where(at_control, lower_param.value, hi)
                else:
                    lo = np.where(at_control, np.minimum(lo, lower_param.value), lo)
                    hi = np.where(at_control, np.maximum(hi, lower_param.value), hi)
            # `np.array`: a 0-d result of the arithmetic is a numpy scalar, which cannot be
            # frozen.
            return Prediction(
                severity=x,
                estimate=np.array(estimate),
                lo=np.array(lo),
                hi=np.array(hi),
                level=level,
            )

        assert self.params is not None and self.covariance is not None  # status is OK
        mu = self.params["mu"].value
        s = self.params["s"].value
        upper_param = self.params["upper"]
        lower_param = self.params["lower"]
        upper = upper_param.value
        lower = lower_param.value
        covariance = self.covariance

        z_score = float(norm.ppf(0.5 + level / 2.0))

        # x = 0 on a log axis is not special-cased here: both band functions below route it
        # through the exact same logit-P gradient/delta computation as every other severity
        # so `predict(0)` and `predict(1e-12)` agree, and a *fixed* `upper` still gets a
        # zero-width band there for free (perturbing any other estimated parameter cannot move
        # a constant), with no separate formula to keep in sync.
        if upper_param.fixed and lower_param.fixed:
            estimate, lo, hi = _predict_band_fixed_asymptotes(
                x,
                mu=mu,
                s=s,
                upper=upper,
                lower=lower,
                scale=self.axis.scale,
                link=self.link,
                covariance=covariance,
                z_score=z_score,
            )
        else:
            estimate, lo, hi = _predict_band_estimated_asymptote(
                x,
                params=self.params,
                scale=self.axis.scale,
                link=self.link,
                covariance=covariance,
                z_score=z_score,
            )

        return Prediction(severity=x, estimate=estimate, lo=lo, hi=hi, level=level)

    def _predict_gaussian(self, x: NDArray[np.float64], *, level: float) -> Prediction:
        """:meth:`predict` for a Gaussian fit, ``x`` already validated."""
        if self.direction == "increasing":
            band = _mirror_fit(self).predict(x, level=level)
            return Prediction(
                severity=x,
                estimate=np.array(-band.estimate),
                lo=np.array(-band.hi),
                hi=np.array(-band.lo),
                level=level,
            )
        assert self.params is not None and self.covariance is not None
        assert self.residual is not None  # status is OK
        mu, s = self.params["mu"].value, self.params["s"].value
        upper, lower = self.params["upper"].value, self.params["lower"].value
        flat = x.reshape(-1)
        control = (flat == 0.0) if self.axis.scale == "log" else np.zeros(flat.size, dtype=bool)
        t = np.zeros(flat.size, dtype=np.float64)
        t[~control] = _transform(flat[~control], scale=self.axis.scale)
        with np.errstate(all="ignore"):
            z = np.where(control, 0.0, (t - mu) / s)
            cdf = np.where(control, 0.0, _f_cdf(z, self.link))
            survival = np.where(control, 1.0, _f_survival(z, self.link))
            density = np.where(control, 0.0, _f_pdf(z, self.link))
        gradient = {
            "mu": (upper - lower) * density / s,
            "s": (upper - lower) * density * z / s,
            "upper": survival,
            "lower": cdf,
        }
        columns = np.column_stack([gradient[name] for name in self.covariance.names])
        matrix = np.array(self.covariance.matrix, dtype=np.float64)
        variance = np.einsum("ij,jk,ik->i", columns, matrix, columns)
        half = float(student_t.ppf(0.5 + level / 2.0, self.residual.df)) * np.sqrt(
            np.maximum(variance, 0.0)
        )
        estimate = np.where(control, upper, upper * survival + lower * cdf)
        return Prediction(
            severity=x,
            estimate=np.array(estimate.reshape(x.shape)),
            lo=np.array((estimate - half).reshape(x.shape)),
            hi=np.array((estimate + half).reshape(x.shape)),
            level=level,
        )


# ================================================================================================
# The mirror between the two directions
# ================================================================================================

_ASYMPTOTE_NAMES = frozenset({"upper", "lower"})
_SWAPPED_PARAMETER_NAME = {"mu": "mu", "s": "s", "upper": "lower", "lower": "upper"}


def _mirror_fit(fit: Fit) -> Fit:
    """The same fit seen as the complementary outcome, with the direction flipped.

    ``P' = 1 - P`` turns ``P = l + (u - l) * F(z)`` into ``P' = u' - (u' - l') * F(z)`` with
    ``u' = 1 - l`` and ``l' = 1 - u``, so an increasing fit and the decreasing fit of the
    complement counts share ``mu``, ``s``, the link, the log-likelihood and the status. The
    asymptotes are swapped and complemented, each ``fixed`` flag follows its asymptote, and the
    covariance exchanges the names ``upper`` and ``lower`` and changes the sign of every entry
    that couples exactly one asymptote to something else (the map's Jacobian is diagonal, with
    entries ``+1`` for ``mu`` and ``s`` and ``-1`` for the asymptotes). Cell counts are
    complemented, so the map applied twice returns the fit it was given, up to rounding.
    Diagnostics, monotonicity, cluster ids and warnings are carried over as they are.

    A Gaussian fit mirrors by negation instead: ``y -> -y`` turns ``P = l + (u - l) * F(z)`` into
    ``P' = u' - (u' - l') * F(z)`` with ``u' = -l`` and ``l' = -u``. Negation is exact in floating
    point, so the map applied twice returns the fit it was given exactly, and the residual sum of
    squares, the degrees of freedom and the log-likelihood are unchanged.
    """
    cells: tuple[Cell | ContinuousCell, ...]
    if fit.model == "gaussian":
        cells = tuple(
            ContinuousCell(severity=c.severity, n=c.n, mean=-c.mean, sd=c.sd)
            for c in fit.cells
            if isinstance(c, ContinuousCell)
        )
    else:
        cells = tuple(
            Cell(severity=c.severity, successes=c.trials - c.successes, trials=c.trials)
            for c in fit.cells
            if isinstance(c, Cell)
        )
    negate = fit.model == "gaussian"
    direction = "decreasing" if fit.direction == "increasing" else "increasing"
    params: dict[str, Parameter] | None = None
    covariance: Covariance | None = None
    if fit.params is not None:
        assert fit.covariance is not None  # status is OK
        upper, lower = fit.params["upper"], fit.params["lower"]
        params = {
            "mu": fit.params["mu"],
            "s": fit.params["s"],
            "upper": Parameter(
                value=-lower.value if negate else 1.0 - lower.value, fixed=lower.fixed
            ),
            "lower": Parameter(
                value=-upper.value if negate else 1.0 - upper.value, fixed=upper.fixed
            ),
        }
        names = fit.covariance.names
        matrix = tuple(
            tuple(
                -entry if (a in _ASYMPTOTE_NAMES) != (b in _ASYMPTOTE_NAMES) else entry
                for b, entry in zip(names, row, strict=True)
            )
            for a, row in zip(names, fit.covariance.matrix, strict=True)
        )
        covariance = Covariance(
            names=tuple(_SWAPPED_PARAMETER_NAME[n] for n in names), matrix=matrix
        )
    return replace(fit, direction=direction, params=params, covariance=covariance, cells=cells)


@overload
def _mirror_observations(obs: Observations) -> Observations: ...
@overload
def _mirror_observations(obs: ContinuousObservations) -> ContinuousObservations: ...
def _mirror_observations(
    obs: Observations | ContinuousObservations,
) -> Observations | ContinuousObservations:
    """The complementary outcome of ``obs``, for the other direction: ``successes`` becomes
    ``trials - successes`` on every row, with everything else (cluster ids included) unchanged.
    A continuous series is negated instead (every row's mean changes sign, its spread does not)."""
    if isinstance(obs, ContinuousObservations):
        return ContinuousObservations(
            axis=obs.axis,
            outcome=obs.outcome,
            direction="decreasing" if obs.direction == "increasing" else "increasing",
            severity=obs.severity,
            n=obs.n,
            mean=tuple(-m for m in obs.mean),
            within_ss=obs.within_ss,
            cluster=obs.cluster,
        )
    return Observations(
        axis=obs.axis,
        outcome=obs.outcome,
        direction="decreasing" if obs.direction == "increasing" else "increasing",
        severity=obs.severity,
        successes=tuple(n - k for k, n in zip(obs.successes, obs.trials, strict=True)),
        trials=obs.trials,
        cluster=obs.cluster,
    )


# ================================================================================================
# Curve evaluation and the full binomial log-likelihood (shared by both fitting paths and by
# Fit.predict, so a reported log_likelihood and a predicted curve always agree with each other).
# ================================================================================================


def _transform(x: NDArray[np.float64], *, scale: str) -> NDArray[np.float64]:
    """``log x`` on a log axis, ``x`` unchanged on a linear axis. Never
    called on an exact-zero severity on a log axis -- callers route that case around this
    function entirely, so this never evaluates ``log(0)``.
    """
    if scale == "log":
        return np.asarray(np.log(x), dtype=np.float64)
    return x


def _f_cdf(z: NDArray[np.float64], link: str) -> NDArray[np.float64]:
    """The link's CDF, modelling the probability of *failure* rising with ``z``. ``cloglog``'s
    ``1 - exp(-exp(z))`` is written as ``-expm1(-exp(z))`` for numerical stability, not for a
    different value.
    """
    if link == "probit":
        return np.asarray(norm.cdf(z), dtype=np.float64)
    if link == "logit":
        return np.asarray(expit(z), dtype=np.float64)
    return np.asarray(-np.expm1(-np.exp(z)), dtype=np.float64)  # link == "cloglog"


def _f_survival(z: NDArray[np.float64], link: str) -> NDArray[np.float64]:
    """``1 - _f_cdf(z, link)``, computed directly from each link's own survival identity --
    never as ``1 - _f_cdf(z, link)`` -- so it stays numerically accurate at the tail where that
    subtraction would round to exactly ``0.0``: ``scipy.stats.norm.sf`` for probit;
    ``expit(-z) == 1 - expit(z)`` exactly for logit (the logistic CDF's own reflection identity);
    ``exp(-exp(z)) == 1 - (-expm1(-exp(z)))`` exactly for cloglog (``_f_cdf``'s own algebraic
    identity, read the other way). ``_f_cdf`` itself is already accurate where *it* is small (large
    negative ``z``); this is its mirror image, accurate where *it* is close to ``1`` (large positive
    ``z``).
    """
    if link == "probit":
        return np.asarray(norm.sf(z), dtype=np.float64)
    if link == "logit":
        return np.asarray(expit(-z), dtype=np.float64)
    return np.asarray(np.exp(-np.exp(z)), dtype=np.float64)  # link == "cloglog"


def _curve_success_probability(
    severity: ArrayLike,
    *,
    mu: float,
    s: float,
    upper: float,
    lower: float,
    scale: str,
    link: str,
    center: float = 0.0,
    spread: float = 1.0,
) -> NDArray[np.float64]:
    """``P(x) = u - (u - l) * F(z)``, with the log-axis zero-severity
    control handled directly: ``P(0) = u`` exactly, never through ``log(0)``.

    ``center``/``spread`` let a caller evaluate this on the *centred-and-scaled* transformed
    covariate ``t' = (t - center) / spread`` instead of ``t`` itself (:func:`_centering_and_scale`):
    ``z = (t' - mu) / s`` there, so ``mu``/``s`` are then on that same rescaled units, not
    natural ones. Defaulted to the identity (``0.0``, ``1.0``), so every existing caller that
    evaluates the curve at its final, natural-scale ``mu``/``s`` is unaffected.
    """
    x = np.asarray(severity, dtype=np.float64)
    result = np.empty_like(x)
    zero_control = (scale == "log") & (x == 0.0)
    nonzero = ~zero_control
    if np.any(nonzero):
        t = (_transform(x[nonzero], scale=scale) - center) / spread
        z = (t - mu) / s
        result[nonzero] = upper - (upper - lower) * _f_cdf(z, link)
    if np.any(zero_control):
        result[zero_control] = upper
    return result


def _curve_success_and_failure_probability(
    severity: ArrayLike,
    *,
    mu: float,
    s: float,
    upper: float,
    lower: float,
    scale: str,
    link: str,
    center: float = 0.0,
    spread: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(p, 1 - p)`` for the curve at ``severity``, each computed *directly* rather than one as
    the complement of the other, so both stay numerically accurate at their own informative tail
    -- used by :func:`robustmargin.diagnose`'s goodness-of-fit sums, which need both ``p`` and
    ``1 - p`` (and their product) without either rounding to an exact ``0``/``1`` for a cell that is
    not :func:`robustmargin.diagnose`'s own *structural* exclusion.

    ``p = l + (u - l) * (1 - F(z))``, using :func:`_f_survival` -- accurate where ``p`` itself is
    small (high severity, under ``direction="decreasing"``), the tail where :func:`_f_cdf` alone
    would already have rounded to exactly ``1`` and so silently rounded ``p`` to exactly ``l``.
    ``1 - p = (1 - u) + (u - l) * F(z)``, using :func:`_f_cdf` directly -- accurate where it is
    small (low severity), the tail :func:`_f_cdf` is already accurate on. Neither value is
    clamped here (that is the caller's job, against whatever floor it needs); the two need not
    sum to exactly ``1.0`` in floating point, by design -- forcing that would reintroduce the
    exact cancellation this function exists to avoid.

    Shares :func:`_curve_success_probability`'s log-axis zero-severity-control handling (``p(0)
    = u`` exactly) and ``center``/``spread`` reparametrisation -- see that function's
    docstring for both.
    """
    x = np.asarray(severity, dtype=np.float64)
    p = np.empty_like(x)
    q = np.empty_like(x)
    zero_control = (scale == "log") & (x == 0.0)
    nonzero = ~zero_control
    if np.any(nonzero):
        t = (_transform(x[nonzero], scale=scale) - center) / spread
        z = (t - mu) / s
        p[nonzero] = lower + (upper - lower) * _f_survival(z, link)
        q[nonzero] = (1.0 - upper) + (upper - lower) * _f_cdf(z, link)
    if np.any(zero_control):
        p[zero_control] = upper
        q[zero_control] = 1.0 - upper
    return p, q


def _count_cells(fit: Fit) -> tuple[Cell, ...]:
    """The pooled counts of a binomial ``fit``, narrowed from ``Fit.cells``' two possible types.
    Every binomial-only code path reads its cells through this, so a Gaussian fit that reached
    one by mistake stops here instead of failing on a missing attribute."""
    cells = tuple(c for c in fit.cells if isinstance(c, Cell))
    if len(cells) != len(fit.cells):
        raise ValueError(
            f"this operation needs the pooled counts of a binomial fit, got model={fit.model!r}"
        )
    return cells


def _cells_to_arrays(
    cells: tuple[Cell, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    severity = np.array([c.severity for c in cells], dtype=np.float64)
    successes = np.array([float(c.successes) for c in cells], dtype=np.float64)
    trials = np.array([float(c.trials) for c in cells], dtype=np.float64)
    return severity, successes, trials


def _transformed_span(severity: NDArray[np.float64], *, scale: str) -> float:
    """The span (max - min) of the *likelihood-contributing* transformed levels: a log-axis
    zero-severity control is excluded, since it carries no covariate value at all and so
    cannot be part of a "tested range" on the transformed axis. Used by
    ``_FLAT_SLOPE_SPAN_FACTOR``'s flat-curve check. Guaranteed non-zero whenever
    ``_check_separation`` has already passed, because that check requires at least two distinct
    likelihood-contributing severities.
    """
    nonzero = severity > 0.0 if scale == "log" else np.ones_like(severity, dtype=bool)
    transformed = _transform(severity[nonzero], scale=scale)
    if transformed.size == 0:
        return 0.0
    return float(np.max(transformed) - np.min(transformed))


def _is_positive_definite(matrix: NDArray[np.float64]) -> bool:
    """:class:`Covariance`'s positive-definiteness rule, shared so the fitters can check a
    natural-scale covariance or information matrix *before* building one: a failure then gives
    ``Status.NOT_CONVERGED`` with a reason instead of an uncaught ``ValueError``, or a
    ``LinAlgError`` from :func:`numpy.linalg.eigvalsh` on a non-finite matrix. Symmetry is not
    checked here: :func:`numpy.linalg.eigvalsh` reads the lower triangle only, and the fitters
    symmetrise their own matrices before calling this.

    The test is on the correlation matrix ``D^-1/2 . matrix . D^-1/2``: finite, every diagonal
    entry positive, and minimum eigenvalue ``> 1e-12``. A natural-scale covariance mixes
    severity-unit parameters (``mu``, ``s``, whose variances scale with the square of the
    caller's units) with probabilities (``upper``, ``lower``); a rule on the raw eigenvalues
    would make the fit's status depend on severity units. Since, for a positive semi-definite
    ``matrix``, ``min eig(correlation) >= min eig(matrix) / max(diag(matrix))``, every matrix a
    raw-eigenvalue rule (``> 1e-12 * max(1, max|entry|)``) would accept is accepted here too.
    """
    if not bool(np.all(np.isfinite(matrix))):
        return False
    diagonal = np.diag(matrix)
    if not bool(np.all(diagonal > 0.0)):
        return False
    scale = np.sqrt(diagonal)
    correlation = matrix / np.outer(scale, scale)
    # No symmetrising here: eigvalsh reads the lower triangle only, so a matrix within
    # Covariance's symmetry tolerance is judged on its lower-triangle entries.
    return float(np.min(np.linalg.eigvalsh(correlation))) > _POSITIVE_DEFINITE_RELATIVE_TOLERANCE


def _centering_and_scale(severity: NDArray[np.float64], *, scale: str) -> tuple[float, float]:
    """The centring constant ``m`` and scale ``d`` used to condition the transformed covariate
    *before* either fitting path ever sees it: fitting ``t' = (t - m) / d`` instead of ``t``
    directly is an exact reparametrisation of the identical curve
    (``mu = m + d * mu'``, ``s = d * s'``; the asymptotes are untouched by it), so it is
    consistent with never hand-rolling a fitter -- it changes the optimiser's working units, not
    the model. Without it, severity units that are very large or very small on the transformed axis
    (for example a linear axis measured in micro-units, or in millions of units) make the
    optimiser's own internal scale blow up or collapse, even though the natural-scale fit itself is
    perfectly well identified.

    ``m`` is the mean and ``d`` the standard deviation of the *likelihood-contributing*
    transformed levels -- the same subset :func:`_transformed_span` and
    :func:`_check_separation` use: a log-axis zero-severity control is excluded, since it
    carries no covariate value at all. Falls back to the span, then to ``1.0``, in the
    degenerate case of a zero standard deviation (should not arise once
    :func:`_check_separation` has passed, since that requires at least two distinct
    likelihood-contributing severities).
    """
    nonzero = severity > 0.0 if scale == "log" else np.ones_like(severity, dtype=bool)
    transformed = _transform(severity[nonzero], scale=scale)
    if transformed.size == 0:
        return 0.0, 1.0
    m = float(np.mean(transformed))
    d = float(np.std(transformed))
    if not d > 0.0:
        span = float(np.max(transformed) - np.min(transformed))
        d = span if span > 0.0 else 1.0
    return m, d


def _f_log_cdf(z: NDArray[np.float64], link: str) -> NDArray[np.float64]:
    """``log F(z)``, computed in log space so it stays finite and accurate where ``F`` is
    far below the smallest double: ``norm.logcdf`` for probit, ``log_expit(z)`` for logit, and
    ``log(-expm1(-exp(z)))`` for cloglog.
    """
    if link == "probit":
        return np.asarray(norm.logcdf(z), dtype=np.float64)
    if link == "logit":
        return np.asarray(log_expit(z), dtype=np.float64)
    # For z < -30, `-expm1(-exp(z))` is exp(z) to double precision and its log is z; evaluating
    # the log directly would underflow to -inf near z = -745.
    with np.errstate(over="ignore", divide="ignore"):
        direct = np.log(-np.expm1(-np.exp(z)))
    return np.asarray(np.where(z < -30.0, z, direct), dtype=np.float64)


def _f_log_survival(z: NDArray[np.float64], link: str) -> NDArray[np.float64]:
    """``log(1 - F(z))``, the mirror of :func:`_f_log_cdf`: ``norm.logsf`` for probit,
    ``log_expit(-z)`` for logit, and ``-exp(z)`` for cloglog.
    """
    if link == "probit":
        return np.asarray(norm.logsf(z), dtype=np.float64)
    if link == "logit":
        return np.asarray(log_expit(-z), dtype=np.float64)
    with np.errstate(over="ignore"):
        return np.asarray(-np.exp(z), dtype=np.float64)


def _log_or_neg_inf(value: float) -> float:
    return math.log(value) if value > 0.0 else -math.inf


def _curve_log_success_and_failure_probability(
    severity: ArrayLike,
    *,
    mu: float,
    s: float,
    upper: float,
    lower: float,
    scale: str,
    link: str,
    center: float = 0.0,
    spread: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(log p, log(1 - p))`` for ``p = l + (u - l) * (1 - F(z))``, each combined in log space:
    ``log p = logaddexp(log l, log(u - l) + log(1 - F))`` and
    ``log(1 - p) = logaddexp(log(1 - u), log(u - l) + log F)``. Neither term is formed as the
    complement of the other, so neither rounds to ``log(0)`` in a tail where the curve is
    smaller than the smallest double. ``log 0`` is ``-inf``, which ``logaddexp`` absorbs.

    Shares :func:`_curve_success_probability`'s zero-severity-control handling (``p(0) = u``)
    and ``center``/``spread`` reparametrisation.
    """
    x = np.asarray(severity, dtype=np.float64)
    log_p = np.empty_like(x)
    log_q = np.empty_like(x)
    zero_control = (scale == "log") & (x == 0.0)
    nonzero = ~zero_control
    log_lower = _log_or_neg_inf(lower)
    log_one_minus_upper = _log_or_neg_inf(1.0 - upper)
    # `l < u` holds only in exact arithmetic; a zero range is the constant curve `p = u`.
    log_range = _log_or_neg_inf(upper - lower)
    if np.any(nonzero):
        t = (_transform(x[nonzero], scale=scale) - center) / spread
        z = (t - mu) / s
        # A non-finite `z` (an optimizer step that overflowed `s`) gives NaN, as it always did;
        # the optimizers and the finiteness guard handle it, so the warning is not repeated here.
        with np.errstate(invalid="ignore"):
            log_p[nonzero] = np.logaddexp(log_lower, log_range + _f_log_survival(z, link))
            log_q[nonzero] = np.logaddexp(log_one_minus_upper, log_range + _f_log_cdf(z, link))
    if np.any(zero_control):
        log_p[zero_control] = _log_or_neg_inf(upper)
        log_q[zero_control] = log_one_minus_upper
    return log_p, log_q


def _full_binomial_log_likelihood(
    severity: NDArray[np.float64],
    successes: NDArray[np.float64],
    trials: NDArray[np.float64],
    *,
    mu: float,
    s: float,
    upper: float,
    lower: float,
    scale: str,
    link: str,
    center: float = 0.0,
    spread: float = 1.0,
) -> float:
    """The full binomial log-likelihood, including the ``lchoose`` combinatorial term, so it
    matches R ``drc``'s ``logLik()`` on both fitting paths (see
    https://github.com/robustmargin/robustmargin/blob/main/tools/drc_reference/README.md).

    Evaluated in log space through :func:`_curve_log_success_and_failure_probability`, so a cell
    deep in a tail keeps a finite, accurate term instead of the ``-inf`` that ``log`` of a
    probability rounded to exactly ``0`` gives. A 0-successes or 0-failures cell has a term of
    exactly ``0.0``, even where the matching log probability is ``-inf``.

    ``center``/``spread`` are forwarded to the curve evaluation -- see there.
    Defaulted to the identity, so this still computes the reported ``Fit.log_likelihood`` on
    natural-scale ``mu``/``s`` against raw severity when a caller omits them (both fitting
    paths' final report does); only the optimisers' own iterations, which
    work in centred-and-scaled units, pass them explicitly.

    Summed with the plain Python builtin ``sum`` over a fixed-order list, not ``numpy.sum``:
    ``numpy.sum``'s pairwise reduction can regroup its additions differently when the array's
    length changes by one, which would make an exact-zero-contribution cell (an all-success
    zero-severity control under a fixed ``upper=1.0``) change the *other* cells' floating-point
    rounding even though its own contribution is exactly ``0.0``. A left-to-right Python sum
    does not have that failure mode: inserting an exact ``0.0`` term anywhere in a strictly
    sequential accumulation is a bit-exact no-op. This is what keeps the aggregation property
    test's "with control" and "without control" fits bit-identical, not merely close.
    """
    log_p, log_q = _curve_log_success_and_failure_probability(
        severity,
        mu=mu,
        s=s,
        upper=upper,
        lower=lower,
        scale=scale,
        link=link,
        center=center,
        spread=spread,
    )
    failures = trials - successes
    lchoose = gammaln(trials + 1.0) - gammaln(successes + 1.0) - gammaln(failures + 1.0)
    with np.errstate(invalid="ignore"):
        success_terms = np.where(successes > 0.0, successes * log_p, 0.0)
        failure_terms = np.where(failures > 0.0, failures * log_q, 0.0)
    terms = lchoose + success_terms + failure_terms
    return float(sum(terms.tolist()))


def _covariance_entry(covariance: Covariance, name_a: str, name_b: str) -> float:
    i = covariance.names.index(name_a)
    j = covariance.names.index(name_b)
    return float(covariance.matrix[i][j])


# ================================================================================================
# Fit.predict's two band scales (documented in full on Fit.predict).
# ================================================================================================


def _predict_band_fixed_asymptotes(
    x: NDArray[np.float64],
    *,
    mu: float,
    s: float,
    upper: float,
    lower: float,
    scale: str,
    link: str,
    covariance: Covariance,
    z_score: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Delta method on the linear predictor ``z``, mapped through ``P = u - (u - l) * F(z)``.
    Only valid when both asymptotes are fixed constants, which is what keeps the resulting band
    inside ``(l, u)`` for every severity: ``F`` maps into ``[0, 1]`` regardless of how uncertain
    ``z`` is, and ``u``, ``l`` themselves carry no uncertainty here.

    A log-axis zero-severity control is handled directly (``estimate = lo = hi = upper``, a
    zero-width band, since ``u`` is a constant here): ``z`` is undefined at ``x = 0`` on a log
    axis, so those entries are routed around the ``z``-based computation entirely rather
    than computed through ``log(0)``.
    """
    estimate = np.empty_like(x)
    lo = np.empty_like(x)
    hi = np.empty_like(x)

    zero_control = (scale == "log") & (x == 0.0)
    nonzero = ~zero_control
    if np.any(zero_control):
        estimate[zero_control] = upper
        lo[zero_control] = upper
        hi[zero_control] = upper

    if np.any(nonzero):
        x_nonzero = x[nonzero]
        z = (_transform(x_nonzero, scale=scale) - mu) / s
        var_mu = _covariance_entry(covariance, "mu", "mu")
        var_s = _covariance_entry(covariance, "s", "s")
        cov_mu_s = _covariance_entry(covariance, "mu", "s")
        var_z = (var_mu + (z**2) * var_s - 2.0 * z * cov_mu_s) / (s**2)
        se_z = np.sqrt(np.clip(var_z, 0.0, None))

        estimate[nonzero] = upper - (upper - lower) * _f_cdf(z, link)
        # F is non-decreasing, so the smaller z-side gives the larger P (P falls as z rises).
        hi[nonzero] = upper - (upper - lower) * _f_cdf(z - z_score * se_z, link)
        lo[nonzero] = upper - (upper - lower) * _f_cdf(z + z_score * se_z, link)
    return estimate, lo, hi


def _predict_band_estimated_asymptote(
    x: NDArray[np.float64],
    *,
    params: Mapping[str, Parameter],
    scale: str,
    link: str,
    covariance: Covariance,
    z_score: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Delta method on ``logit(P)``, using the gradient of ``logit(P(x))`` with respect to
    every *estimated* natural parameter (``covariance.names``, always ``mu``, ``s`` and
    whichever of ``lower``/``upper`` are not fixed). Guarantees the band stays inside
    ``(0, 1)`` regardless of how uncertain an estimated asymptote is, because ``expit`` is used
    to back-transform rather than a linear step on ``P`` itself.
    """
    names = covariance.names
    natural_vector = np.array([params[name].value for name in names], dtype=np.float64)
    fixed_lower = params["lower"].value if params["lower"].fixed else None
    fixed_upper = params["upper"].value if params["upper"].fixed else None
    matrix = np.array([[float(v) for v in row] for row in covariance.matrix], dtype=np.float64)

    def _probability_at(natural: NDArray[np.float64], severity_value: float) -> float:
        mapping = dict(zip(names, natural, strict=True))
        mu = mapping["mu"]
        s = mapping["s"]
        lower = mapping["lower"] if "lower" in mapping else fixed_lower
        upper = mapping["upper"] if "upper" in mapping else fixed_upper
        assert lower is not None and upper is not None
        return float(
            _curve_success_probability(
                np.array([severity_value]),
                mu=mu,
                s=s,
                upper=upper,
                lower=lower,
                scale=scale,
                link=link,
            )[0]
        )

    # A flat, at-least-1-D working view: a 0-d `x` (Fit.predict called with a bare scalar) has
    # no positional index (`x[0]` raises `IndexError` on a 0-d array), so integer-indexed
    # iteration needs a 1-D view regardless of `x`'s own shape. The result is reshaped back to
    # `x.shape` at the end, so a 0-d input still gives 0-d output arrays (Prediction's own
    # documented behaviour), not a shape promoted to `(1,)`.
    flat_x = np.atleast_1d(x)
    estimate = np.empty_like(flat_x)
    lo = np.empty_like(flat_x)
    hi = np.empty_like(flat_x)
    for i in range(flat_x.size):
        severity_value = float(flat_x[i])
        p0 = _probability_at(natural_vector, severity_value)

        def _logit_p(
            natural: NDArray[np.float64], sv: float = severity_value
        ) -> NDArray[np.float64]:
            return np.array([float(_logit(_probability_at(natural, sv)))])

        gradient = _central_difference_jacobian(_logit_p, natural_vector)[0]
        var_logit_p = float(gradient @ matrix @ gradient)
        se_logit_p = math.sqrt(max(var_logit_p, 0.0))
        logit_p0 = float(_logit(p0))
        estimate[i] = p0
        lo[i] = expit(logit_p0 - z_score * se_logit_p)
        hi[i] = expit(logit_p0 + z_score * se_logit_p)
    return estimate.reshape(x.shape), lo.reshape(x.shape), hi.reshape(x.shape)


def _central_difference_jacobian(
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: NDArray[np.float64],
    *,
    relative_step: float = 1e-6,
) -> NDArray[np.float64]:
    """The Jacobian of ``func`` at ``x`` by central finite differences: the delta method's
    numerical-differentiation step (used both to transform a fit's optimizer-scale covariance
    onto the natural scale, and inside :func:`Fit.predict`'s estimated-asymptote band), not a
    fitting procedure. The package never hand-rolls an optimizer or IRLS loop; a numerical
    derivative of an already-fitted quantity is not one.
    """
    x = np.asarray(x, dtype=np.float64)
    f0 = np.asarray(func(x), dtype=np.float64)
    jacobian = np.zeros((f0.size, x.size), dtype=np.float64)
    for j in range(x.size):
        step = relative_step * max(1.0, abs(float(x[j])))
        x_plus = x.copy()
        x_plus[j] += step
        x_minus = x.copy()
        x_minus[j] -= step
        jacobian[:, j] = (
            np.asarray(func(x_plus), dtype=np.float64) - np.asarray(func(x_minus), dtype=np.float64)
        ) / (2.0 * step)
    return jacobian


# ================================================================================================
# fit_dose_response: the estimated-asymptote (GenericLikelihoodModel) path
# ================================================================================================


def _unpack_optimizer_params(
    params: NDArray[np.float64],
    *,
    estimate_lower: bool,
    estimate_upper: bool,
    lower_value: float,
    upper_value: float,
) -> tuple[float, float, float, float]:
    """Map the optimizer's internal ``(mu, log_s, [a], [b])`` vector
    to the natural scale ``(mu, s, lower, upper)``. Whichever of ``a``/``b`` corresponds to a
    *fixed* asymptote is simply absent from ``params`` ("fixed asymptotes are left out of the
    parameter vector"). ``l < u`` holds by construction in every combination:

    - Both estimated: ``l = expit(a)`` and ``u = l + (1 - l) * expit(b)``.
    - ``upper`` estimated, ``lower`` fixed: ``u = l + (1 - l) * expit(b)`` with the fixed ``l``.
    - ``lower`` estimated, ``upper`` fixed: ``l = u * expit(a)``
      with the fixed ``u`` -- *not* the unconstrained ``l = expit(a)`` used when ``upper`` is
      also estimated, which can exceed a fixed ``u`` and converge onto an increasing curve
      (``Fit`` would then reject ``lower.value < upper.value`` at construction). Anchoring ``l``
      to a fraction of the fixed ``u`` keeps ``l`` inside ``(0, u)`` for every finite ``a``.
    """
    mu = float(params[0])
    s = float(np.exp(params[1]))
    idx = 2
    if estimate_lower:
        a_value = float(params[idx])
        idx += 1
        lower = float(upper_value * expit(a_value)) if not estimate_upper else float(expit(a_value))
    else:
        lower = lower_value
    if estimate_upper:
        upper = float(lower + (1.0 - lower) * expit(params[idx]))
        idx += 1
    else:
        upper = upper_value
    return mu, s, lower, upper


@dataclass(frozen=True)
class _GenericLogLikelihood:
    """The estimated-asymptote likelihood, passed to
    ``statsmodels.base.model.GenericLikelihoodModel`` as its ``loglike=`` callable
    (specifying a likelihood for a statsmodels optimizer, never a subclass that reimplements the
    optimizer itself). Held as a plain callable object, not a ``GenericLikelihoodModel`` subclass,
    so nothing here subclasses a type statsmodels ships with no ``py.typed`` marker.

    ``center``/``spread`` are the :func:`_centering_and_scale` constants: the optimizer's own
    ``mu``/``log_s`` therefore live on the *centred-and-scaled* transformed covariate, not on
    natural units, which is what keeps the optimizer's working scale independent of the caller's
    severity units (a fit must not depend on them). ``a``/``b`` (asymptote logits) are untouched by
    this -- only ``mu``/``s`` are rescaled.
    """

    severity: NDArray[np.float64]
    successes: NDArray[np.float64]
    trials: NDArray[np.float64]
    scale: str
    link: str
    estimate_lower: bool
    estimate_upper: bool
    lower_value: float
    upper_value: float
    center: float
    spread: float

    def __call__(self, params: NDArray[np.float64]) -> float:
        mu, s, lower, upper = _unpack_optimizer_params(
            params,
            estimate_lower=self.estimate_lower,
            estimate_upper=self.estimate_upper,
            lower_value=self.lower_value,
            upper_value=self.upper_value,
        )
        return _full_binomial_log_likelihood(
            self.severity,
            self.successes,
            self.trials,
            mu=mu,
            s=s,
            upper=upper,
            lower=lower,
            scale=self.scale,
            link=self.link,
            center=self.center,
            spread=self.spread,
        )

    def batch_log_likelihood(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """The log-likelihood at every row of ``params`` (shape ``(m, p)``, the optimizer's own
        ``(mu, log_s, [a], [b])`` chart), as a vector of ``m`` values.

        It follows :meth:`__call__` term for term: the same chart, the same log-space curve
        (``log p`` and ``log(1 - p)`` from the log survival and log cdf of the link), the
        zero-severity control of a log axis pinned at ``p = u``, and the binomial
        coefficients. A row whose value is not finite there is not finite here. The terms are
        summed in a vectorised order, so a value can differ from :meth:`__call__` in the last
        bits."""
        points = np.asarray(params, dtype=np.float64)
        mu = points[:, 0]
        with np.errstate(over="ignore"):
            s = np.exp(points[:, 1])
        column = 2
        if self.estimate_lower:
            share = expit(points[:, column])
            column += 1
            lower = share if self.estimate_upper else self.upper_value * share
        else:
            lower = np.full(mu.shape, self.lower_value)
        if self.estimate_upper:
            upper = lower + (1.0 - lower) * expit(points[:, column])
        else:
            upper = np.full(mu.shape, self.upper_value)

        def _log_or_neg_inf_array(values: NDArray[np.float64]) -> NDArray[np.float64]:
            positive = values > 0.0
            return np.where(positive, np.log(np.where(positive, values, 1.0)), -np.inf)

        log_lower = _log_or_neg_inf_array(lower)[:, None]
        log_one_minus_upper = _log_or_neg_inf_array(1.0 - upper)[:, None]
        log_range = _log_or_neg_inf_array(upper - lower)[:, None]
        zero_control = (self.scale == "log") & (self.severity == 0.0)
        nonzero = ~zero_control
        log_p = np.empty((mu.size, self.severity.size))
        log_q = np.empty((mu.size, self.severity.size))
        with np.errstate(all="ignore"):
            if np.any(nonzero):
                t = (_transform(self.severity[nonzero], scale=self.scale) - self.center) / (
                    self.spread
                )
                z = (t[None, :] - mu[:, None]) / s[:, None]
                log_p[:, nonzero] = np.logaddexp(
                    log_lower, log_range + _f_log_survival(z, self.link)
                )
                log_q[:, nonzero] = np.logaddexp(
                    log_one_minus_upper, log_range + _f_log_cdf(z, self.link)
                )
            if np.any(zero_control):
                log_p[:, zero_control] = _log_or_neg_inf_array(upper)[:, None]
                log_q[:, zero_control] = log_one_minus_upper
            failures = self.trials - self.successes
            lchoose = (
                gammaln(self.trials + 1.0) - gammaln(self.successes + 1.0) - gammaln(failures + 1.0)
            )
            success_terms = np.where(self.successes > 0.0, self.successes * log_p, 0.0)
            failure_terms = np.where(failures > 0.0, failures * log_q, 0.0)
            total = (lchoose + success_terms + failure_terms).sum(axis=1)
        return np.asarray(total, dtype=np.float64)


def _pooled_rate(successes: NDArray[np.float64], trials: NDArray[np.float64]) -> float:
    return float(np.sum(successes) / np.sum(trials))


def _binomial_group_log_likelihood(
    successes: NDArray[np.float64], trials: NDArray[np.float64], p: float
) -> float:
    """``sum k log p + (n - k) log(1 - p)`` with ``0 log 0 = 0`` (no binomial coefficient)."""
    return float(np.sum(xlogy(successes, p) + xlog1py(trials - successes, -p)))


def _flat_supremum_log_likelihood(
    severity: NDArray[np.float64],
    successes: NDArray[np.float64],
    trials: NDArray[np.float64],
    *,
    estimate_lower: bool,
    estimate_upper: bool,
    lower_value: float,
    upper_value: float,
    scale: str,
) -> float:
    """The supremum of the full binomial log-likelihood (binomial coefficients included) on the
    ``s -> infinity`` boundary, where the curve is flat at a single probability ``p``.

    Without a zero-severity control, or on a linear axis, ``p`` is the pooled rate clipped to the
    feasible constants: ``[l, u]`` with both asymptotes fixed, ``[l, 1]`` with ``upper`` estimated,
    ``[0, u]`` with ``lower`` estimated and ``[0, 1]`` with both estimated.

    On a log axis the zero-severity control is pinned at the upper asymptote ``u`` for every
    parameter value, so only the nonzero levels share ``p``. Write ``low`` for ``l`` when ``lower``
    is fixed and 0 when it is estimated. With ``u`` fixed, the control contributes its
    log-likelihood at ``u`` and the nonzero levels sit at their pooled rate clipped to
    ``[low, u]``. With ``u`` estimated, ``p`` is the pooled nonzero rate floored at ``low``; if the
    control's rate is at least ``p`` then ``u`` is the control's rate, and otherwise
    ``u = p`` is the pooled rate of all levels floored at ``low``.
    """
    low = 0.0 if estimate_lower else lower_value
    coefficients = float(
        np.sum(gammaln(trials + 1.0) - gammaln(successes + 1.0) - gammaln(trials - successes + 1.0))
    )
    control = (severity == 0.0) if scale == "log" else np.zeros(severity.shape, dtype=bool)
    if not bool(np.any(control)):
        high = 1.0 if estimate_upper else upper_value
        p = min(max(_pooled_rate(successes, trials), low), high)
        return coefficients + _binomial_group_log_likelihood(successes, trials, p)

    control_k, control_n = successes[control], trials[control]
    other_k, other_n = successes[~control], trials[~control]
    if estimate_upper:
        p = max(_pooled_rate(other_k, other_n), low)
        u = _pooled_rate(control_k, control_n)
        if u < p:
            u = p = max(_pooled_rate(successes, trials), low)
    else:
        u = upper_value
        p = min(max(_pooled_rate(other_k, other_n), low), u)
    return (
        coefficients
        + _binomial_group_log_likelihood(control_k, control_n, u)
        + _binomial_group_log_likelihood(other_k, other_n, p)
    )


class _StepSupremum(NamedTuple):
    """The step supremum, whether its best configuration is the constant curve (``u == l``), and
    that configuration's asymptotes (``nan`` for both when fewer than two levels give no
    configuration)."""

    value: float
    degenerate: bool
    upper: float
    lower: float


def _step_supremum_log_likelihood(
    severity: NDArray[np.float64],
    successes: NDArray[np.float64],
    trials: NDArray[np.float64],
    *,
    estimate_lower: bool,
    estimate_upper: bool,
    lower_value: float,
    upper_value: float,
    scale: str,
) -> _StepSupremum:
    """The supremum of the full binomial log-likelihood (binomial coefficients included) on the
    ``s -> 0`` boundary of the decreasing curve, where the curve is a step at ``mu``.

    Levels are ordered by severity. For a step between two adjacent levels, the levels before it
    sit at the upper asymptote ``u`` and the levels after it at the lower asymptote ``l``. A single
    level may instead sit on the step itself, with the levels before it at ``u``, the levels after
    it at ``l``, and its own fitted probability its rate clipped to ``[l, u]``; an end level may do
    so only when the asymptote of its empty side is fixed, and a zero-severity control on a log
    axis never does (it always takes ``u``). An
    estimated asymptote is its side's pooled rate subject to ``l <= u`` (when the pooled rates
    violate it, both equal the pooled rate of the two sides together; with ``u`` fixed an
    estimated ``l`` is clipped into ``[0, u]``, and with ``l`` fixed an estimated ``u`` into
    ``[l, 1]``); a fixed asymptote keeps its value. The result is the maximum over all steps and
    free-level choices, with ``degenerate`` true when the best configuration has ``u == l`` (the
    constant curve) and ``upper`` and ``lower`` the best configuration's asymptotes (a fixed
    asymptote keeps its value); fewer than two levels give ``(-inf, False, nan, nan)``.
    """
    order = np.argsort(severity, kind="stable")
    sev = np.asarray(severity, dtype=np.float64)[order]
    k = np.asarray(successes, dtype=np.float64)[order]
    n = np.asarray(trials, dtype=np.float64)[order]
    count = k.size
    if count < 2:
        return _StepSupremum(-math.inf, False, math.nan, math.nan)
    coefficients = float(np.sum(gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)))
    rates = k / n
    has_control = scale == "log" and bool(sev[0] == 0.0)
    everyone = np.arange(count)

    # Each configuration is (baseline-side levels, free level or None, far-side levels). A free
    # level sits at the transition itself, so the sides are the levels before and after it.
    configurations: list[tuple[NDArray[np.intp], int | None, NDArray[np.intp]]] = [
        (everyone[:split], None, everyone[split:]) for split in range(1, count)
    ]
    for level in range(count):
        left, right = everyone[:level], everyone[level + 1 :]
        if has_control and level == 0:
            continue
        if (left.size == 0 and estimate_upper) or (right.size == 0 and estimate_lower):
            continue
        configurations.append((left, level, right))

    best = _StepSupremum(-math.inf, False, math.nan, math.nan)
    for left, free, right in configurations:
        upper = _pooled_rate(k[left], n[left]) if estimate_upper else upper_value
        lower = _pooled_rate(k[right], n[right]) if estimate_lower else lower_value
        if estimate_upper and estimate_lower:
            if lower > upper:
                upper = lower = _pooled_rate(k, n)
        elif estimate_lower:
            lower = min(lower, upper)
        elif estimate_upper:
            upper = max(upper, lower)
        total = coefficients
        total += _binomial_group_log_likelihood(k[left], n[left], upper)
        total += _binomial_group_log_likelihood(k[right], n[right], lower)
        if free is not None:
            p_free = float(np.clip(rates[free], lower, upper))
            total += _binomial_group_log_likelihood(k[free : free + 1], n[free : free + 1], p_free)
        if total > best.value:
            best = _StepSupremum(total, abs(upper - lower) <= 1e-12, float(upper), float(lower))
    return best


def _pilot_start(
    severity: NDArray[np.float64],
    successes: NDArray[np.float64],
    trials: NDArray[np.float64],
    *,
    scale: str,
    link: str,
    center: float,
    spread: float,
) -> tuple[float, float, float, float, bool]:
    """A starting point for the estimated-asymptote optimizer -- "only a starting point",
    never itself reported as a fit. ``upper``/``lower`` start at the
    control rate (or the maximum rate) and the minimum rate (probabilities, unaffected by
    ``center``/``spread``); ``mu``/``s`` start from a GLM pilot fit of the *rescaled* rates
    against the same link, on the same centred-and-scaled covariate
    ``t' = (t - center) / spread`` (:func:`_centering_and_scale`) the real optimizer fits on --
    both far better conditioned than fitting the raw 0/1-bounded rates on raw severity units
    when the true asymptotes are not 0/1 and/or the severity units are far from order 1.
    ``mu0``/``s0`` are therefore returned already on that same rescaled ("primed") scale, ready
    to seed the optimizer's own ``mu'``/``log_s'`` directly with no further conversion.

    The pilot GLM itself is never skipped or wrapped in a try/except: a
    `PerfectSeparationWarning` from it is filtered to "ignore", not treated as a failure, so the
    only two things that actually change what is returned are (a) fewer than two distinct
    nonzero severities, when there is nothing to regress against at all, and (b) a wrong-sign
    pilot slope (``beta1 <= 0``) -- either keeps the neutral default (``mu0' = 0``, ``s0' = 1``,
    both sensible defaults *because* the rescaled covariate is centred at 0 with spread 1):
    robustness matters more than precision for a value the real optimizer only starts from.
    Case (b) is also reported back via ``pilot_wrong_sign``, so a caller whose real fit later
    fails can attach the wrong-sign reason too.

    Returns
    -------
    tuple[float, float, float, float, bool]
        ``(mu0, s0, upper_start, lower_start, pilot_wrong_sign)``, with ``mu0``/``s0`` on the
        centred-and-scaled covariate. ``pilot_wrong_sign`` is ``True`` only when the pilot GLM
        actually ran (at least two distinct nonzero severities) and its fitted slope was
        ``<= 0``.
    """
    rate = successes / trials
    control_mask = severity == 0.0
    if scale == "log" and bool(np.any(control_mask)):
        upper_start = float(rate[control_mask][0])
    else:
        upper_start = float(np.max(rate))
    lower_start = float(np.min(rate))
    if not upper_start > lower_start:
        upper_start, lower_start = 1.0, 0.0

    nonzero = severity > 0.0 if scale == "log" else np.ones_like(severity, dtype=bool)
    transformed_scaled = (_transform(severity[nonzero], scale=scale) - center) / spread
    mu0 = float(np.median(transformed_scaled)) if transformed_scaled.size else 0.0
    s0 = 1.0
    pilot_wrong_sign = False

    if transformed_scaled.size and len(np.unique(severity[nonzero])) >= 2:
        eps = 1e-3
        rescaled_failure_rate = np.clip(
            (upper_start - rate[nonzero]) / (upper_start - lower_start), eps, 1.0 - eps
        )
        exog = add_constant(transformed_scaled)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PerfectSeparationWarning)
            pilot = GLM(
                rescaled_failure_rate,
                exog,
                family=families.Binomial(link=_STATSMODELS_LINKS[link]),
                var_weights=trials[nonzero],
            ).fit()
        beta0 = float(pilot.params[0])
        beta1 = float(pilot.params[1])
        # Start values only: a wrong-sign pilot keeps the neutral (median, 1.0) start.
        if beta1 > 0.0:
            mu0 = -beta0 / beta1
            s0 = 1.0 / beta1
        else:
            pilot_wrong_sign = True
    return mu0, s0, upper_start, lower_start, pilot_wrong_sign


def _boundary_warning(
    final_params: NDArray[np.float64],
    *,
    estimate_lower: bool,
    estimate_upper: bool,
    lower: float,
    upper: float,
    mirrored: bool = False,
) -> str | None:
    """An estimated asymptote pinned to a parameter-space boundary,
    checked both on the optimizer's own logit-scale parameter (``a``/``b``, ``|.| > 15``) and
    on the natural-scale probability itself (within ``1e-6`` of 0 or 1). Either test alone can
    miss a boundary fit the other catches (a huge logit-scale value that still rounds to a
    natural-scale probability inside the epsilon, or vice versa via the numerics of a
    particular optimizer step), so both are checked.

    With ``mirrored`` the arguments describe the complement of an increasing outcome, and the
    message names the asymptote of the outcome as the caller supplied it: the roles of ``lower``
    and ``upper`` are exchanged, and a probability is reported as ``1 - p``.
    """
    idx = 2
    if estimate_lower:
        a_value = float(final_params[idx])
        idx += 1
        if abs(a_value) > _ASYMPTOTE_LOGIT_BOUND:
            return _logit_boundary_message(
                "upper" if mirrored else "lower", a_value, mirrored=mirrored
            )
    if estimate_upper:
        b_value = float(final_params[idx])
        if abs(b_value) > _ASYMPTOTE_LOGIT_BOUND:
            return _logit_boundary_message(
                "lower" if mirrored else "upper", b_value, mirrored=mirrored
            )
    if estimate_lower and estimate_upper and lower < _ASYMPTOTE_PROBABILITY_EPS:
        name, value, comparison = (
            ("upper", 1.0 - lower, f"> 1 - {_ASYMPTOTE_PROBABILITY_EPS}")
            if mirrored
            else ("lower", lower, f"< {_ASYMPTOTE_PROBABILITY_EPS}")
        )
        return (
            f"fit_dose_response: the estimated {name} asymptote reached the probability "
            f"boundary ({name}={value!r} {comparison}) -- no interior maximum exists"
        )
    if estimate_lower and not estimate_upper and (upper - lower) < _ASYMPTOTE_PROBABILITY_EPS:
        # With a fixed upper, l = u * expit(a) ranges over (0, u). l -> 0 (a -> -inf) is *not*
        # exempt from a boundary check here -- the |a| > 15 test above already applies to it, the
        # same as it would for any other estimated asymptote's logit parameter, so it is not
        # "ordinary" merely because there is no separate probability-floor test for it. This
        # eps-based test instead covers the *other* end, l -> u (a -> +inf, expit(a) -> 1): the
        # degenerate property there is the gap (u - l) collapsing (the curve going flat, u == l, no
        # dynamic range at all), which a fixed magnitude bound on `a` alone does not directly test,
        # because how large `a` needs to be for u - l to fall below eps depends on the fixed u
        # itself.
        estimated, fixed = ("upper", "lower") if mirrored else ("lower", "upper")
        return (
            f"fit_dose_response: the estimated {estimated} asymptote reached the probability "
            f"boundary near the fixed {fixed} asymptote (upper - lower = {upper - lower!r} < "
            f"{_ASYMPTOTE_PROBABILITY_EPS}) -- no interior maximum exists"
        )
    if estimate_upper and upper > 1.0 - _ASYMPTOTE_PROBABILITY_EPS:
        name, value, comparison = (
            ("lower", 1.0 - upper, f"< {_ASYMPTOTE_PROBABILITY_EPS}")
            if mirrored
            else ("upper", upper, f"> 1 - {_ASYMPTOTE_PROBABILITY_EPS}")
        )
        return (
            f"fit_dose_response: the estimated {name} asymptote reached the probability "
            f"boundary ({name}={value!r} {comparison}) -- no interior maximum exists"
        )
    return None


_NONFINITE_LOG_LIKELIHOOD_WARNING = (
    "fit_dose_response: the log-likelihood at the fitted parameters is not finite -- no "
    "estimate is reported"
)
_STEP_PLATEAU_WARNING = (
    "fit_dose_response: no fitted curve exceeded the log-likelihood of a step between adjacent "
    "tested levels (s -> 0) -- the data do not identify the slope"
)
_NONFINITE_ESTIMATE_WARNING = (
    "fit_dose_response: the optimizer's estimate or its log-likelihood is not finite -- no "
    "estimate is reported"
)
_REPLACED = "so it was replaced by direct likelihood maximisation of the same model"
_REFIT_NOT_CONVERGED_WARNING = f"fit_dose_response: the GLM did not converge, {_REPLACED}"
_REFIT_OPTIMALITY_WARNING = (
    f"fit_dose_response: the GLM estimate failed the optimality check (it is not a maximum of "
    f"the likelihood), {_REPLACED}"
)
_REFIT_INFORMATION_WARNING = (
    f"fit_dose_response: the GLM's information could not be evaluated, {_REPLACED}"
)
_INFORMATION_RECOVERED_WARNING = (
    "fit_dose_response: the GLM's observed information could not be evaluated (it is not "
    "finite at cells where the fitted probability is 0 or 1), so it was computed from the "
    "stable log-likelihood instead"
)


def _logit_boundary_message(name: str, value: float, *, mirrored: bool) -> str:
    """The boundary warning for an asymptote whose optimizer-scale logit parameter ran past
    ``_ASYMPTOTE_LOGIT_BOUND``. The parameter's value is on the optimizer's own scale and is
    reported only for a fit made in the caller's own orientation: for an increasing outcome the
    optimizer's parameter belongs to the complement and is left out."""
    detail = (
        f"|logit-scale parameter| > {_ASYMPTOTE_LOGIT_BOUND}"
        if mirrored
        else f"logit-scale {'a' if name == 'lower' else 'b'}={value!r}, "
        f"|{'a' if name == 'lower' else 'b'}| > {_ASYMPTOTE_LOGIT_BOUND}"
    )
    return (
        f"fit_dose_response: the estimated {name} asymptote reached the parameter "
        f"boundary ({detail}) -- no interior maximum exists"
    )


def _failure_warnings(reason: str, *, pilot_wrong_sign: bool, mirrored: bool) -> tuple[str, ...]:
    """The ``warnings`` tuple for a ``NOT_CONVERGED`` estimated-asymptote fit: ``reason`` alone,
    or ``reason`` plus the shared wrong-sign/flat reason when a pilot GLM slope on the (rescaled)
    data had the wrong sign for the direction -- unless ``reason`` already *is* that
    shared reason, in which case it is not duplicated.
    """
    wrong_sign = _wrong_sign_or_flat_warning(mirrored=mirrored)
    if pilot_wrong_sign and reason != wrong_sign:
        return (reason, wrong_sign)
    return (reason,)


def _step_plateau_reason(
    log_likelihood: float, step: _StepSupremum, *, mirrored: bool
) -> str | None:
    """The reason when a fitted log-likelihood does not exceed the step supremum by more than the
    optimality tolerance, else ``None``. When the best step is the constant curve, the data are
    flat or rising and the reason is the shared wrong-sign/flat one.
    """
    if log_likelihood > step.value + _NEWTON_DECREMENT_TOLERANCE:
        return None
    return (
        _wrong_sign_or_flat_warning(mirrored=mirrored) if step.degenerate else _STEP_PLATEAU_WARNING
    )


# Starting (mu, s) on the centred-and-scaled covariate for the extra BFGS runs of `_fit_generic`;
# each keeps the pilot's asymptote starts. Deterministic and independent of severity units.
_ALTERNATIVE_STARTS: tuple[tuple[float, float], ...] = (
    (0.0, 1.0),
    (-1.0, 0.25),
    (1.0, 0.25),
    (-1.0, 4.0),
    (1.0, 4.0),
)


# The scan that seeds a further search from the likelihood's whole surface. ``log(s)`` runs from
# ``log(span) - _SCAN_LOG_S_BELOW`` to ``log(span) + _SCAN_LOG_S_ABOVE`` in steps of
# ``_SCAN_LOG_S_STEP``, with ``span`` the range of the finite tested levels: a transition squeezed
# between two levels has ``s`` orders of magnitude below the span, a flat curve a few above. A free
# asymptote coordinate takes the shares ``_SCAN_SHARES`` of its range and the fit's own value. With
# a single estimated asymptote the value of the step supremum's best configuration is one more share
# (a shallow maximum lies next to the step plateau's asymptotes, between grid shares). With both
# estimated, the step configuration is added only as the pairs (step, step), (step, fit) and
# (fit, step), since the upper share depends on the lower asymptote and a crossed grid would
# rarely land on the step's upper value. Shares are kept ``_FIT_SHARE_EDGE`` from the ends of the
# range and expressed as a share in the optimizer's chart.
_SCAN_LOG_S_BELOW = math.log(1.0e8)
_SCAN_LOG_S_ABOVE = math.log(1.0e3)
_SCAN_LOG_S_STEP = 0.1
_SCAN_SHARES = (0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99)
_FIT_SHARE_EDGE = 1.0e-6
# The location of the scan, on the centred-and-scaled covariate. Step positions put the location at
# a tested level, at the midpoint of adjacent levels and one unit beyond each end. Tail positions
# put it at ``mean(levels) + s * w`` for a multiple ``w`` of the scale: the few far values
# ``_SCAN_TAIL_FAR`` and every ``_SCAN_TAIL_STEP`` over ``[-_SCAN_TAIL_RANGE, _SCAN_TAIL_RANGE]``.
# These reach a location in a gap between a pinned zero-severity control and the first nonzero
# level, which a location inside the tested range cannot.
_SCAN_TAIL_FAR = (-40.0, -20.0, -10.0, 10.0, 20.0, 40.0)
_SCAN_TAIL_RANGE = 8.0
_SCAN_TAIL_STEP = 0.25
# The scan's best points, ranked by the profile in ``log(s)``, that start the search.
_SCAN_STARTS = 4


def _scan_log_s_grid(span: float) -> NDArray[np.float64]:
    low = math.log(span) - _SCAN_LOG_S_BELOW
    high = math.log(span) + _SCAN_LOG_S_ABOVE
    count = math.floor((high - low) / _SCAN_LOG_S_STEP) + 1
    return low + _SCAN_LOG_S_STEP * np.arange(count, dtype=np.float64)


def _share_logits(fit_share: float, extra_shares: Sequence[float] = ()) -> list[float]:
    """The logits of the scanned shares of a free coordinate's range: :data:`_SCAN_SHARES`, the
    fit's own share and any ``extra_shares``, each kept :data:`_FIT_SHARE_EDGE` from the ends of the
    range. A non-finite extra share is ignored; duplicates are removed."""
    shares = (fit_share, *(share for share in extra_shares if math.isfinite(share)))
    clipped = [float(np.clip(share, _FIT_SHARE_EDGE, 1.0 - _FIT_SHARE_EDGE)) for share in shares]
    return [float(_logit(share)) for share in dict.fromkeys((*clipped, *_SCAN_SHARES))]


def _step_shares(objective: _GenericLogLikelihood, step: _StepSupremum) -> tuple[float, float]:
    """The shares, in the optimizer's chart, of the step supremum's best configuration for the
    lower and the upper asymptote (``nan`` where the share is undefined).

    With both asymptotes estimated the lower share is ``l`` and the upper share
    ``(u - l) / (1 - l)``; with only ``lower`` estimated its share is ``l / u``, and with only
    ``upper`` estimated the upper share is ``(u - l) / (1 - l)``
    with the fixed ``l``."""
    upper, lower = step.upper, step.lower
    if objective.estimate_upper:
        upper_share = (upper - lower) / (1.0 - lower) if lower < 1.0 else math.nan
    else:
        upper_share = math.nan
    if not objective.estimate_lower:
        lower_share = math.nan
    elif objective.estimate_upper:
        lower_share = lower
    else:
        lower_share = lower / upper if upper > 0.0 else math.nan
    return float(lower_share), float(upper_share)


def _top_local_maxima(profile: NDArray[np.float64], count: int) -> list[int]:
    """Indices into ``profile`` of its highest local maxima, best first, at most ``count``.

    A non-finite entry counts as ``-inf``, and the two ends of the array have a neighbour of
    ``-inf`` beyond them. A local maximum is a maximal run of equal consecutive values, finite and
    strictly higher than the entry before it and the entry after it; the run is represented by
    its first (lowest) index, so a plateau is one maximum and an end point can be one. Ties in
    value are ordered by index."""
    values = np.where(np.isfinite(profile), profile, -np.inf)
    maxima: list[int] = []
    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and values[stop + 1] == values[start]:
            stop += 1
        before = values[start - 1] if start > 0 else -np.inf
        after = values[stop + 1] if stop + 1 < values.size else -np.inf
        if values[start] > -np.inf and values[start] > before and values[start] > after:
            maxima.append(start)
        start = stop + 1
    maxima.sort(key=lambda index: -values[index])
    return maxima[:count]


def _scan_starts(
    objective: _GenericLogLikelihood,
    selected_params: NDArray[np.float64],
    step: _StepSupremum,
) -> list[NDArray[np.float64]]:
    """Starting points for a search beyond the pilot and the fixed alternative starts, found by a
    scan that uses no optimizer and no random numbers.

    The scan evaluates :meth:`_GenericLogLikelihood.batch_log_likelihood` over ``log(s)`` (from
    eight orders of magnitude below the span of the tested levels to three above, in steps of
    :data:`_SCAN_LOG_S_STEP`), the location (step positions and tail positions, see
    :data:`_SCAN_TAIL_FAR`) and each estimated asymptote's coordinate (the logits of
    :func:`_share_logits`, which include the share of the value in ``selected_params``, the
    parameters of the result selected so far). The share, in the optimizer's chart, of the
    asymptote of ``step``, the best configuration of the step supremum, is one more share when a
    single asymptote is estimated; when both are, it is added as the rows pairing the step's
    shares with each other and with the fit's, after the product of the two coordinates' shares
    (none where a share is undefined). The best value at each ``log(s)`` is the profile in
    ``log(s)``, and the points that attain its :data:`_SCAN_STARTS` highest local maxima are
    returned, best first, as parameter vectors in the optimizer's chart. Fewer maxima give fewer
    starts, and none if no point has a finite value.

    This reaches a maximum that a start inside the tested range cannot, such as one between a
    pinned zero-severity control and the first nonzero level, or a narrow basin."""
    nonzero = (
        objective.severity > 0.0
        if objective.scale == "log"
        else np.ones(objective.severity.shape, dtype=bool)
    )
    levels = np.unique(
        (_transform(objective.severity[nonzero], scale=objective.scale) - objective.center)
        / objective.spread
    )
    span = float(levels[-1] - levels[0]) if levels.size > 1 else 1.0
    span = span if span > 0.0 else 1.0
    middles = (levels[:-1] + levels[1:]) / 2.0
    steps = np.concatenate([levels, middles, [levels[0] - 1.0, levels[-1] + 1.0]])
    tail = np.unique(
        np.concatenate(
            [
                np.asarray(_SCAN_TAIL_FAR),
                np.arange(-_SCAN_TAIL_RANGE, _SCAN_TAIL_RANGE + _SCAN_TAIL_STEP, _SCAN_TAIL_STEP),
            ]
        )
    )
    centre = float(np.mean(levels))

    extras: list[list[float]] = []
    fit_logits: list[float] = []
    step_logits: list[float | None] = []
    column = 2
    both = objective.estimate_lower and objective.estimate_upper
    for estimated, step_share in zip(
        (objective.estimate_lower, objective.estimate_upper),
        _step_shares(objective, step),
        strict=True,
    ):
        if estimated:
            fit_share = float(expit(selected_params[column]))
            extras.append(_share_logits(fit_share, () if both else (step_share,)))
            fit_logits.append(extras[-1][0])
            step_logits.append(
                float(_logit(float(np.clip(step_share, _FIT_SHARE_EDGE, 1.0 - _FIT_SHARE_EDGE))))
                if math.isfinite(step_share)
                else None
            )
            column += 1
    rows_list = list(itertools.product(*extras))
    if both:
        (a_fit, b_fit), (a_step, b_step) = fit_logits, step_logits
        pairs = (
            ([(a_step, b_step)] if a_step is not None and b_step is not None else [])
            + ([(a_step, b_fit)] if a_step is not None else [])
            + ([(a_fit, b_step)] if b_step is not None else [])
        )
        rows_list.extend(dict.fromkeys(pair for pair in pairs if pair not in set(rows_list)))
    asymptotes = (
        np.array(rows_list, dtype=np.float64) if extras else np.empty((1, 0), dtype=np.float64)
    )

    log_s_grid = _scan_log_s_grid(span)
    best_value = np.full(log_s_grid.size, -np.inf)
    best_point: list[NDArray[np.float64]] = []
    for index, log_s in enumerate(log_s_grid):
        s = math.exp(log_s)
        locations = np.concatenate([steps, centre + s * tail])
        rows = np.empty((locations.size * asymptotes.shape[0], 2 + len(extras)))
        rows[:, 0] = np.repeat(locations, asymptotes.shape[0])
        rows[:, 1] = log_s
        rows[:, 2:] = np.tile(asymptotes, (locations.size, 1))
        values = objective.batch_log_likelihood(rows)
        values = np.where(np.isfinite(values), values, -np.inf)
        best = int(np.argmax(values))
        best_value[index] = values[best]
        best_point.append(rows[best])
    return [
        np.array(best_point[index], dtype=np.float64)
        for index in _top_local_maxima(best_value, _SCAN_STARTS)
    ]


def _fit_generic(
    obs: Observations,
    *,
    cells: tuple[Cell, ...],
    cluster_ids: tuple[str | int, ...] | None,
    link: str,
    upper_is_estimate: bool,
    upper_value: float,
    lower_is_estimate: bool,
    lower_value: float,
    mirrored: bool = False,
) -> Fit:
    """The estimated-or-non-trivial-fixed-asymptote path: ``statsmodels``
    ``GenericLikelihoodModel`` given :class:`_GenericLogLikelihood`, fit with ``method="bfgs"``
    then a ``method="newton"`` polish -- both statsmodels optimizers,
    never a hand-rolled Newton or IRLS loop.

    BFGS runs from the pilot start and from each of ``_ALTERNATIVE_STARTS`` (fixed starting
    points on the centred-and-scaled covariate, no random numbers), unless both asymptotes are
    fixed at exactly 1 and 0. There the log-likelihood is concave in ``(-mu/s, 1/s)`` for the
    probit, logit and complementary log-log links, and ``(mu, log s)`` is a one-to-one
    reparametrisation of the slope ``1/s > 0``, so every local maximum is global.
    The pilot run is polished by Newton's method; an alternative start replaces it only when its
    BFGS log-likelihood beats the pilot's best by more than the optimality tolerance, and is then
    polished and checked in its place. Unless both asymptotes are fixed at 1 and 0, a scan of
    the likelihood (:func:`_scan_starts`) then seeds BFGS from its best points, and a start that
    beats the selected result by more than the same tolerance replaces it in turn. A fit whose
    log-likelihood does not exceed the supremum on either boundary (a step as ``s -> 0``, a flat
    curve as ``s -> infinity``) is ``NOT_CONVERGED``.

    ``mirrored`` says that ``obs`` is the complement of an increasing outcome: it changes only
    the wording of a failure's reason, so that it names the asymptotes and direction of the
    outcome the caller supplied.
    """
    axis = obs.axis

    # Drop a structurally zero-contribution zero-severity control row before building the
    # optimizer's own data: with `upper` fixed at exactly 1.0 on a log axis,
    # `_curve_success_probability`'s own zero-severity-control rule pins this cell's fitted
    # probability to exactly `upper` for every possible mu/s/lower, so its log-likelihood
    # contribution is an exact constant with respect to every parameter this path estimates -- by
    # the time `_fit_generic` is reached with upper fixed at 1.0, `_check_control_incompatible` has
    # already ruled out a control with any failures here, so that constant is exactly 0. Leaving
    # that row in the data had two distinct effects on the fit, not one, and this exclusion removes
    # both at once: (1) `_pilot_start`'s own `upper_start` reads the *control's* rate when one is
    # present (here, exactly 1.0) rather than the maximum rate among the other levels (here, about
    # 0.96), which moved the pilot `mu0`/`s0` the real optimizer started from; (2)
    # `GenericLikelihoodModel.fit` (statsmodels' own optimizer, not one of ours) minimises
    # `-loglike(params) / nobs`, where `nobs` is the row count it was built with, so a
    # zero-contribution row changes `nobs` and, with it, the scale at which `gtol`/`tol` judge
    # convergence, even though the likelihood function itself is unchanged at every parameter value.
    # Excluding it here mirrors `_fit_glm`'s own `fit_cells` exclusion below, and keeps both the
    # pilot start and `nobs` identical to a fit over the same data with no control row at all. An
    # *estimated* upper is never exempt: the zero_control branch then reports `upper` itself, a free
    # parameter, so the control genuinely identifies it and must stay in the optimizer's data.
    fit_cells = (
        tuple(c for c in cells if c.severity != 0.0)
        if axis.scale == "log" and not upper_is_estimate and upper_value == 1.0
        else cells
    )
    severity, successes, trials = _cells_to_arrays(fit_cells)

    # Centre and scale the transformed covariate before either the pilot or the real optimiser
    # ever sees it (a fit must not depend on the caller's choice of severity
    # units). This is an exact reparametrisation, not a different model -- see
    # _centering_and_scale and _GenericLogLikelihood.
    center, spread = _centering_and_scale(severity, scale=axis.scale)

    mu0, s0, u0, l0, pilot_wrong_sign = _pilot_start(
        severity, successes, trials, scale=axis.scale, link=link, center=center, spread=spread
    )

    start = [mu0, math.log(s0)]
    eps = 1e-3
    lower_for_start = l0 if lower_is_estimate else lower_value
    upper_for_start = u0 if upper_is_estimate else upper_value
    if lower_is_estimate:
        if upper_is_estimate:
            a0 = float(_logit(np.clip(lower_for_start, eps, 1.0 - eps)))
        else:
            # l = u * expit(a) when upper is fixed, so a's start value inverts *that* formula with
            # the fixed u, not l = expit(a) directly.
            a0 = float(_logit(np.clip(lower_for_start / upper_for_start, eps, 1.0 - eps)))
        start.append(a0)
    if upper_is_estimate:
        denominator = max(1.0 - lower_for_start, 1e-6)
        fraction = float(np.clip((upper_for_start - lower_for_start) / denominator, eps, 1.0 - eps))
        start.append(float(_logit(fraction)))
    start_params = np.array(start, dtype=np.float64)

    loglike_fn = _GenericLogLikelihood(
        severity=severity,
        successes=successes,
        trials=trials,
        scale=axis.scale,
        link=link,
        estimate_lower=lower_is_estimate,
        estimate_upper=upper_is_estimate,
        lower_value=lower_value,
        upper_value=upper_value,
        center=center,
        spread=spread,
    )
    exog = np.ones((severity.size, start_params.size))
    model = GenericLikelihoodModel(successes, exog, loglike=loglike_fn)

    # When the data are a step between adjacent levels the log-likelihood has no finite maximum
    # (it rises as s -> 0), whatever the optimizer reported: a fit that has not exceeded the
    # closed-form supremum on the s -> 0 boundary is a point on that plateau. The supremum is
    # taken over the same observations and fixed values the objective uses.
    step_supremum = _step_supremum_log_likelihood(
        severity,
        successes,
        trials,
        estimate_lower=lower_is_estimate,
        estimate_upper=upper_is_estimate,
        lower_value=lower_value,
        upper_value=upper_value,
        scale=axis.scale,
    )
    flat_supremum = _flat_supremum_log_likelihood(
        severity,
        successes,
        trials,
        estimate_lower=lower_is_estimate,
        estimate_upper=upper_is_estimate,
        lower_value=lower_value,
        upper_value=upper_value,
        scale=axis.scale,
    )

    # The BFGS stage is only ever a starting point for the Newton polish: it routinely reports its
    # own `converged=False` / ConvergenceWarning even when it has landed close enough for Newton to
    # finish the job, so only the *final* (Newton) result's own convergence signal decides the
    # status below -- checking a warning merged across both stages would treat that expected
    # intermediate state as a failure.
    def _bfgs(from_params: NDArray[np.float64]) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.fit(
                start_params=from_params, method="bfgs", disp=False, maxiter=2000, gtol=1e-10
            )

    def _polish(from_bfgs: Any) -> tuple[Any, list[warnings.WarningMessage]]:
        # A singular Hessian makes the Newton step raise; that is reported as `None`.
        try:
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                polished = model.fit(
                    start_params=np.asarray(from_bfgs.params, dtype=np.float64),
                    method="newton",
                    disp=False,
                    maxiter=200,
                    tol=1e-10,
                )
        except np.linalg.LinAlgError:
            return None, []
        return polished, record

    def _finite_llf(fitted: Any) -> float:
        value = float(fitted.llf)
        return value if math.isfinite(value) else -math.inf

    bfgs_result = _bfgs(start_params)
    result, caught = _polish(bfgs_result)
    pilot_best = _finite_llf(bfgs_result)
    if result is not None:
        pilot_best = max(pilot_best, _finite_llf(result))

    # The log-likelihood can have more than one interior local maximum once an asymptote is
    # estimated or fixed away from (1, 0), and BFGS from the pilot start can settle in a basin that
    # is not the global one. BFGS is therefore also run from fixed alternative starts on the
    # centred-and-scaled covariate. Only if the best of them beats the pilot's best log-likelihood
    # by more than the optimality tolerance does that start replace the pilot (after its own Newton
    # polish); every check below then judges it, and a failure of those checks is reported, never
    # replaced by the lower pilot maximum. With both asymptotes fixed at (1, 0) the
    # log-likelihood is concave in (b0, b1) = (-mu/s, 1/s) for the probit, logit and cloglog links
    # (F and 1 - F are log-concave), and (mu, log s) is a one-to-one reparametrisation of b1 > 0,
    # so every local maximum is global.
    concave = (
        not lower_is_estimate
        and not upper_is_estimate
        and upper_value == 1.0
        and (lower_value == 0.0)
    )
    if not concave:
        best_alternative: Any = None
        best_alternative_llf = -math.inf
        for alt_mu, alt_s in _ALTERNATIVE_STARTS:
            alt_start = start_params.copy()
            alt_start[0] = alt_mu
            alt_start[1] = math.log(alt_s)
            alt_bfgs = _bfgs(alt_start)
            if _finite_llf(alt_bfgs) > best_alternative_llf:
                best_alternative, best_alternative_llf = alt_bfgs, _finite_llf(alt_bfgs)
        if best_alternative is not None and best_alternative_llf > (
            pilot_best + _NEWTON_DECREMENT_TOLERANCE
        ):
            bfgs_result = best_alternative
            result, caught = _polish(bfgs_result)

    def _judge(bfgs_result: Any, result: Any, caught: list[warnings.WarningMessage]) -> Fit:
        """The checks that turn an optimizer result into a :class:`Fit` or a refusal."""
        if result is None:
            # A singular Hessian during the Newton polish is usually the symptom of an estimated
            # asymptote drifting to its boundary; name that cause when the BFGS
            # stage already shows it, so the reason does not depend on which platform's linear
            # algebra happens to fail first.
            bfgs_params = np.asarray(bfgs_result.params, dtype=np.float64)
            _, bfgs_s_scaled, bfgs_lower, bfgs_upper = _unpack_optimizer_params(
                bfgs_params,
                estimate_lower=lower_is_estimate,
                estimate_upper=upper_is_estimate,
                lower_value=lower_value,
                upper_value=upper_value,
            )
            singular_reason = _boundary_warning(
                bfgs_params,
                estimate_lower=lower_is_estimate,
                estimate_upper=upper_is_estimate,
                lower=bfgs_lower,
                upper=bfgs_upper,
                mirrored=mirrored,
            )
            if (
                singular_reason is None
                and spread * bfgs_s_scaled
                > _FLAT_SLOPE_SPAN_FACTOR * _transformed_span(severity, scale=axis.scale)
            ):
                singular_reason = _wrong_sign_or_flat_warning(mirrored=mirrored)
            if singular_reason is None:
                singular_reason = _step_plateau_reason(
                    float(bfgs_result.llf), step_supremum, mirrored=mirrored
                )
            if (
                singular_reason is None
                and float(bfgs_result.llf) <= flat_supremum + _NEWTON_DECREMENT_TOLERANCE
            ):
                singular_reason = _wrong_sign_or_flat_warning(mirrored=mirrored)
            if singular_reason is None:
                singular_reason = (
                    "fit_dose_response: the optimizer's Hessian was singular while fitting -- "
                    "no interior maximum was found"
                )
            return _status_only_fit(
                obs,
                cells=cells,
                cluster_ids=cluster_ids,
                link=link,
                status=Status.NOT_CONVERGED,
                fit_warnings=_failure_warnings(
                    singular_reason, pilot_wrong_sign=pilot_wrong_sign, mirrored=mirrored
                ),
            )
        convergence_warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        converged_flag = bool(result.mle_retvals.get("converged", False))
        final_params = np.asarray(result.params, dtype=np.float64)

        estimate_is_finite = bool(np.all(np.isfinite(final_params))) and math.isfinite(
            float(result.llf)
        )
        mu = s = lower = upper = math.nan
        if estimate_is_finite:
            mu_scaled, s_scaled, lower, upper = _unpack_optimizer_params(
                final_params,
                estimate_lower=lower_is_estimate,
                estimate_upper=upper_is_estimate,
                lower_value=lower_value,
                upper_value=upper_value,
            )
            # mu/s from the optimizer are on the centred-and-scaled covariate; map back to natural
            # units (the exact inverse of _centering_and_scale's reparametrisation). lower/upper are
            # already natural -- asymptotes are untouched by the covariate rescaling.
            mu = center + spread * mu_scaled
            s = spread * s_scaled

        # An estimated asymptote at a parameter-space boundary keeps its own reason, even on data
        # that are a step. A non-finite Newton result is judged on the BFGS parameters instead.
        boundary_params = (
            final_params if estimate_is_finite else np.asarray(bfgs_result.params, dtype=np.float64)
        )
        _, _, boundary_lower, boundary_upper = _unpack_optimizer_params(
            boundary_params,
            estimate_lower=lower_is_estimate,
            estimate_upper=upper_is_estimate,
            lower_value=lower_value,
            upper_value=upper_value,
        )
        boundary_message = _boundary_warning(
            boundary_params,
            estimate_lower=lower_is_estimate,
            estimate_upper=upper_is_estimate,
            lower=boundary_lower,
            upper=boundary_upper,
            mirrored=mirrored,
        )
        if boundary_message is not None:
            return _status_only_fit(
                obs,
                cells=cells,
                cluster_ids=cluster_ids,
                link=link,
                status=Status.NOT_CONVERGED,
                fit_warnings=_failure_warnings(
                    boundary_message, pilot_wrong_sign=pilot_wrong_sign, mirrored=mirrored
                ),
            )

        # A flat curve up to rounding, checked before anything that would build a covariance from a
        # near-degenerate slope.
        span = _transformed_span(severity, scale=axis.scale)
        if estimate_is_finite and s > _FLAT_SLOPE_SPAN_FACTOR * span:
            return _status_only_fit(
                obs,
                cells=cells,
                cluster_ids=cluster_ids,
                link=link,
                status=Status.NOT_CONVERGED,
                fit_warnings=_failure_warnings(
                    _wrong_sign_or_flat_warning(mirrored=mirrored),
                    pilot_wrong_sign=pilot_wrong_sign,
                    mirrored=mirrored,
                ),
            )

        # The best log-likelihood any stage reached: a stage that overshot to a non-finite value
        # does not count against the finite ones.
        best_llf = max(
            float(bfgs_result.llf),
            float(result.llf) if math.isfinite(float(result.llf)) else -math.inf,
        )
        plateau_reason = _step_plateau_reason(best_llf, step_supremum, mirrored=mirrored)
        if plateau_reason is None and best_llf <= flat_supremum + _NEWTON_DECREMENT_TOLERANCE:
            # The other boundary: a fit that has not exceeded the flat curve's best
            # log-likelihood is not an interior maximum.
            plateau_reason = _wrong_sign_or_flat_warning(mirrored=mirrored)
        if plateau_reason is not None:
            return _status_only_fit(
                obs,
                cells=cells,
                cluster_ids=cluster_ids,
                link=link,
                status=Status.NOT_CONVERGED,
                fit_warnings=_failure_warnings(
                    plateau_reason, pilot_wrong_sign=pilot_wrong_sign, mirrored=mirrored
                ),
            )

        if not estimate_is_finite:
            return _status_only_fit(
                obs,
                cells=cells,
                cluster_ids=cluster_ids,
                link=link,
                status=Status.NOT_CONVERGED,
                fit_warnings=_failure_warnings(
                    _NONFINITE_ESTIMATE_WARNING,
                    pilot_wrong_sign=pilot_wrong_sign,
                    mirrored=mirrored,
                ),
            )

        # Newton's own `converged` flag only bounds the last step, so the final log-likelihood
        # must not have regressed below the BFGS stage's (relative tolerance -- an absolute
        # tolerance here does not scale with the log-likelihood's own magnitude, which depends on
        # the sample size, not on anything a caller controls), and the score at the final
        # parameters must be near zero.
        bfgs_llf = float(bfgs_result.llf)
        llf_regressed = float(result.llf) < bfgs_llf - _LLF_REGRESSION_TOLERANCE * (
            1.0 + abs(bfgs_llf)
        )
        score = np.asarray(model.score(final_params), dtype=np.float64)
        score_tolerance = _SCORE_TOLERANCE_FACTOR * (1.0 + abs(float(result.llf)))
        score_not_near_zero = bool(np.any(np.abs(score) > score_tolerance))

        # Newton's step tolerance is below the roundoff of the numerical derivatives its step is
        # built from, so the step criterion can stall exactly at the maximum and report
        # non-convergence there. The Newton decrement of the same objective is scale-free, so it
        # decides instead.
        step_stall_at_maximum = False
        if not converged_flag or convergence_warned:
            decrement = _newton_decrement_half(loglike_fn, final_params)
            step_stall_at_maximum = decrement <= _NEWTON_DECREMENT_TOLERANCE

        not_converged_reason: str | None = None
        if (not converged_flag or convergence_warned) and not step_stall_at_maximum:
            not_converged_reason = "fit_dose_response: the optimizer did not converge"
        elif llf_regressed:
            not_converged_reason = (
                "fit_dose_response: the Newton polish's log-likelihood is lower than the BFGS "
                "stage's -- the optimizer did not converge"
            )
        elif score_not_near_zero:
            not_converged_reason = (
                "fit_dose_response: the score at the optimum is not near zero -- the optimizer "
                "did not converge"
            )
        if not_converged_reason is not None:
            return _status_only_fit(
                obs,
                cells=cells,
                cluster_ids=cluster_ids,
                link=link,
                status=Status.NOT_CONVERGED,
                fit_warnings=_failure_warnings(
                    not_converged_reason, pilot_wrong_sign=pilot_wrong_sign, mirrored=mirrored
                ),
            )

        # A non-finite Hessian entry is checked before eigvalsh ever sees it
        # (numpy.linalg.eigvalsh raises LinAlgError on a non-finite array), and the
        # positive-definite check itself is the same one Covariance applies, done here first so a
        # broken Hessian never reaches that constructor.
        hessian = np.asarray(model.hessian(final_params), dtype=np.float64)
        neg_hessian = -(hessian + hessian.T) / 2.0
        if not _is_positive_definite(neg_hessian):
            return _status_only_fit(
                obs,
                cells=cells,
                cluster_ids=cluster_ids,
                link=link,
                status=Status.NOT_CONVERGED,
                fit_warnings=_failure_warnings(
                    "fit_dose_response: the Hessian at the optimum is not finite and "
                    "positive-definite -- no interior maximum was found",
                    pilot_wrong_sign=pilot_wrong_sign,
                    mirrored=mirrored,
                ),
            )

        names: list[str] = ["mu", "s"]
        if lower_is_estimate:
            names.append("lower")
        if upper_is_estimate:
            names.append("upper")

        def _natural_vector(p: NDArray[np.float64]) -> NDArray[np.float64]:
            mu_scaled_, s_scaled_, lower_, upper_ = _unpack_optimizer_params(
                p,
                estimate_lower=lower_is_estimate,
                estimate_upper=upper_is_estimate,
                lower_value=lower_value,
                upper_value=upper_value,
            )
            # Fold the covariate rescaling into the same numeric Jacobian (mu/s only; asymptotes
            # are already natural).
            values = [center + spread * mu_scaled_, spread * s_scaled_]
            if lower_is_estimate:
                values.append(lower_)
            if upper_is_estimate:
                values.append(upper_)
            return np.array(values, dtype=np.float64)

        jacobian = _central_difference_jacobian(_natural_vector, final_params)
        optimizer_cov = np.asarray(result.cov_params(), dtype=np.float64)
        natural_cov = jacobian @ optimizer_cov @ jacobian.T
        natural_cov = (natural_cov + natural_cov.T) / 2.0

        # Covariance's own positive-definiteness rule, on the correlation scale so the status does
        # not depend on severity units.
        if not _is_positive_definite(natural_cov):
            return _status_only_fit(
                obs,
                cells=cells,
                cluster_ids=cluster_ids,
                link=link,
                status=Status.NOT_CONVERGED,
                fit_warnings=_failure_warnings(
                    "fit_dose_response: the natural-scale covariance is not finite and "
                    "positive-definite -- no interior maximum was found",
                    pilot_wrong_sign=pilot_wrong_sign,
                    mirrored=mirrored,
                ),
            )

        # The reported log-likelihood is always the full binomial log-likelihood over every cell
        # (`Fit.cells`, including a control `fit_cells` dropped above) -- `fit_cells`/`severity`
        # etc. above are the optimizer's own reduced view, not what gets reported.
        all_severity, all_successes, all_trials = _cells_to_arrays(cells)
        log_likelihood = _full_binomial_log_likelihood(
            all_severity,
            all_successes,
            all_trials,
            mu=mu,
            s=s,
            upper=upper,
            lower=lower,
            scale=axis.scale,
            link=link,
        )
        if not math.isfinite(log_likelihood):
            return _status_only_fit(
                obs,
                cells=cells,
                cluster_ids=cluster_ids,
                link=link,
                status=Status.NOT_CONVERGED,
                fit_warnings=_failure_warnings(
                    _NONFINITE_LOG_LIKELIHOOD_WARNING,
                    pilot_wrong_sign=pilot_wrong_sign,
                    mirrored=mirrored,
                ),
            )

        params = {
            "mu": Parameter(value=mu, fixed=False),
            "s": Parameter(value=s, fixed=False),
            "lower": Parameter(value=lower, fixed=not lower_is_estimate),
            "upper": Parameter(value=upper, fixed=not upper_is_estimate),
        }
        covariance = Covariance(
            names=tuple(names), matrix=tuple(tuple(row) for row in natural_cov.tolist())
        )

        final_fit = Fit(
            axis=axis,
            outcome=obs.outcome,
            direction=obs.direction,
            model="binomial",
            link=link,
            params=params,
            covariance=covariance,
            log_likelihood=log_likelihood,
            cells=cells,
            status=Status.OK,
            cluster_ids=cluster_ids,
            warnings=(),
            provenance={},
        )
        return final_fit

    if not concave:
        # The starts above lie in the tested range, so a maximum outside it (between a pinned
        # control and the first level, or in a narrow basin) can be missed. The likelihood is
        # scanned once; if BFGS from the scan's best points beats the selected result by more than
        # the optimality tolerance, that point is polished and replaces it. Every check then runs
        # on the selection, with no fallback to the lower maximum.
        selected_best = _finite_llf(bfgs_result)
        if result is not None:
            selected_best = max(selected_best, _finite_llf(result))
        selected_params = np.asarray(
            (result if result is not None else bfgs_result).params, dtype=np.float64
        )
        if not bool(np.all(np.isfinite(selected_params))):
            selected_params = np.asarray(bfgs_result.params, dtype=np.float64)
        if not bool(np.all(np.isfinite(selected_params))):
            selected_params = start_params
        scan_best: Any = None
        scan_best_llf = -math.inf
        for scan_start in _scan_starts(loglike_fn, selected_params, step_supremum):
            scan_bfgs = _bfgs(scan_start)
            if _finite_llf(scan_bfgs) > scan_best_llf:
                scan_best, scan_best_llf = scan_bfgs, _finite_llf(scan_bfgs)
        # A replacement is chosen only when its BFGS value beats the selected result by more than
        # the switch margin. The judgement then refuses it if its Newton polish falls more than
        # 1e-8 * (1 + |llf|) below that BFGS value. A reported fit is therefore never more than
        # that relative tolerance below the best value any stage reached, and a larger drop is a
        # refusal, never a return to the lower result.
        if scan_best is not None and scan_best_llf > selected_best + _NEWTON_DECREMENT_TOLERANCE:
            bfgs_result = scan_best
            result, caught = _polish(bfgs_result)

    return _judge(bfgs_result, result, caught)


# ================================================================================================
# fit_dose_response: the fixed u=1, l=0 path (statsmodels GLM)
# ================================================================================================


# The GLM path and the profile's inner fit accept an estimate only when its Newton decrement on
# the stable log-likelihood, lambda^2 = g' (-H)^-1 g, satisfies lambda^2 / 2 <= this value: the
# log-likelihood the estimate would still gain from a Newton step. Unlike a bound on the raw score,
# it does not depend on the parametrisation or on the sample size.
_NEWTON_DECREMENT_TOLERANCE = 1e-6


_STANDARD_ERROR_STEP_FRACTION = 1e-2


def _numerical_gradient_and_hessian(
    func: Callable[[NDArray[np.float64]], float],
    x: NDArray[np.float64],
    step: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The gradient and Hessian of ``func`` at ``x`` by central differences, with the given
    per-coordinate ``step`` or by default ``1e-4 * max(1, |x_j|)`` -- numerical differentiation
    of an already-fitted quantity, not a fitting procedure.
    """
    x = np.asarray(x, dtype=np.float64)
    size = x.size
    if step is None:
        step = 1e-4 * np.maximum(1.0, np.abs(x))
    f0 = func(x)
    gradient = np.zeros(size, dtype=np.float64)
    hessian = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        up = x.copy()
        up[i] += step[i]
        down = x.copy()
        down[i] -= step[i]
        f_up, f_down = func(up), func(down)
        gradient[i] = (f_up - f_down) / (2.0 * step[i])
        hessian[i, i] = (f_up - 2.0 * f0 + f_down) / step[i] ** 2
        for j in range(i + 1, size):
            corners = []
            for sign_i, sign_j in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                point = x.copy()
                point[i] += sign_i * step[i]
                point[j] += sign_j * step[j]
                corners.append(func(point))
            hessian[i, j] = hessian[j, i] = (corners[0] - corners[1] - corners[2] + corners[3]) / (
                4.0 * step[i] * step[j]
            )
    return gradient, hessian


def _newton_decrement_half(
    func: Callable[[NDArray[np.float64]], float], x: NDArray[np.float64]
) -> float:
    """``lambda^2 / 2`` for the log-likelihood ``func`` at ``x``, with ``lambda^2 = g' (-H)^-1 g``
    from the numerical score and observed information. ``inf`` when the information is not
    finite and positive-definite, which is itself a failed optimality check.

    The finite-difference steps follow the curvature: a first pass with steps proportional to
    ``|x|`` gives the information, and the derivatives are then recomputed with each step a
    fixed small fraction of the coordinate's standard error, so truncation error stays small
    relative to the curvature on badly scaled functions. Steps are only ever reduced: enlarging
    them for a weakly identified coordinate would measure curvature far from the estimate.
    """
    gradient, hessian = _numerical_gradient_and_hessian(func, x)
    information = -(hessian + hessian.T) / 2.0
    if not bool(np.all(np.isfinite(information))) or not bool(np.all(np.isfinite(gradient))):
        return math.inf
    if float(np.min(np.linalg.eigvalsh(information))) <= 0.0:
        return math.inf
    variances = np.diag(np.linalg.inv(information))
    if bool(np.all(np.isfinite(variances))) and bool(np.all(variances > 0.0)):
        default_step = 1e-4 * np.maximum(1.0, np.abs(np.asarray(x, dtype=np.float64)))
        scaled_step = np.minimum(default_step, _STANDARD_ERROR_STEP_FRACTION * np.sqrt(variances))
        gradient, hessian = _numerical_gradient_and_hessian(func, x, step=scaled_step)
        information = -(hessian + hessian.T) / 2.0
        if not bool(np.all(np.isfinite(information))) or not bool(np.all(np.isfinite(gradient))):
            return math.inf
        if float(np.min(np.linalg.eigvalsh(information))) <= 0.0:
            return math.inf
    return float(gradient @ np.linalg.solve(information, gradient)) / 2.0


def _refit_by_direct_likelihood(
    obs: Observations,
    *,
    cells: tuple[Cell, ...],
    cluster_ids: tuple[str | int, ...] | None,
    link: str,
    reason: str,
    mirrored: bool,
) -> Fit:
    """Refit the fixed ``upper=1, lower=0`` model by direct maximisation of its likelihood
    through the estimated-asymptote path with both asymptotes fixed: the same link, the same
    curve, a statsmodels optimizer. That path's own convergence, score and Hessian checks decide
    the status, and the result carries ``reason``, which states which check replaced the GLM
    estimate.
    """
    refit = _fit_generic(
        obs,
        cells=cells,
        cluster_ids=cluster_ids,
        link=link,
        upper_is_estimate=False,
        upper_value=1.0,
        lower_is_estimate=False,
        lower_value=0.0,
        mirrored=mirrored,
    )
    return replace(refit, warnings=(reason, *refit.warnings))


def _fit_glm(
    obs: Observations,
    *,
    cells: tuple[Cell, ...],
    cluster_ids: tuple[str | int, ...] | None,
    link: str,
    mirrored: bool = False,
) -> Fit:
    axis = obs.axis
    fit_cells = tuple(c for c in cells if not (axis.scale == "log" and c.severity == 0.0))
    severity, successes, trials = _cells_to_arrays(fit_cells)
    failures = trials - successes

    # Centre and scale the transformed covariate before the GLM ever sees it (a fit must not
    # depend on the caller's choice of severity units). An exact reparametrisation of the
    # identical model -- see _centering_and_scale.
    center, spread = _centering_and_scale(severity, scale=axis.scale)
    exog = add_constant((_transform(severity, scale=axis.scale) - center) / spread)
    endog = failures / trials

    model = GLM(
        endog, exog, family=families.Binomial(link=_STATSMODELS_LINKS[link]), var_weights=trials
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
        glm_result = model.fit()
    convergence_warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)

    # The GLM's clipped IRLS can report non-convergence, or converge to a point that is not the
    # maximum, on data whose maximum is finite and unique. Either case is refit as the same
    # model by direct likelihood maximisation, and the result says so.
    if convergence_warned or not bool(glm_result.converged):
        return _refit_by_direct_likelihood(
            obs,
            cells=cells,
            cluster_ids=cluster_ids,
            link=link,
            reason=_REFIT_NOT_CONVERGED_WARNING,
            mirrored=mirrored,
        )

    beta0 = float(glm_result.params[0])
    beta1 = float(glm_result.params[1])
    if beta1 <= 0.0:
        return _status_only_fit(
            obs,
            cells=cells,
            cluster_ids=cluster_ids,
            link=link,
            status=Status.NOT_CONVERGED,
            fit_warnings=(_wrong_sign_or_flat_warning(mirrored=mirrored),),
        )

    # beta0/beta1 are on the centred-and-scaled covariate; map back to natural units (the exact
    # inverse of _centering_and_scale's reparametrisation).
    mu_scaled = -beta0 / beta1
    s_scaled = 1.0 / beta1
    mu = center + spread * mu_scaled
    s = spread * s_scaled

    # A slope that is technically positive but effectively flat (for example beta1 ~ 1e-15) makes s
    # explode, which would otherwise pass Covariance's own positive-definite check while being
    # numerically meaningless (s on the order of 1e61).
    span = _transformed_span(severity, scale=axis.scale)
    if s > _FLAT_SLOPE_SPAN_FACTOR * span:
        return _status_only_fit(
            obs,
            cells=cells,
            cluster_ids=cluster_ids,
            link=link,
            status=Status.NOT_CONVERGED,
            fit_warnings=(_wrong_sign_or_flat_warning(mirrored=mirrored),),
        )

    # Optimality of the GLM estimate on the stable log-likelihood, in the GLM's own coefficients:
    # the Newton decrement is the same in any parametrisation.
    def _beta_loglike(beta: NDArray[np.float64]) -> float:
        if not float(beta[1]) > 0.0:
            return -math.inf
        return _full_binomial_log_likelihood(
            severity,
            successes,
            trials,
            mu=-float(beta[0]) / float(beta[1]),
            s=1.0 / float(beta[1]),
            upper=1.0,
            lower=0.0,
            scale=axis.scale,
            link=link,
            center=center,
            spread=spread,
        )

    glm_beta = np.array([beta0, beta1], dtype=np.float64)
    if _newton_decrement_half(_beta_loglike, glm_beta) > _NEWTON_DECREMENT_TOLERANCE:
        return _refit_by_direct_likelihood(
            obs,
            cells=cells,
            cluster_ids=cluster_ids,
            link=link,
            reason=_REFIT_OPTIMALITY_WARNING,
            mirrored=mirrored,
        )

    # Both fitting paths report covariance from the *observed* information
    # (the inverse of the negative Hessian of the log-likelihood at the MLE), not statsmodels'
    # own `cov_params()` (the *expected*/Fisher information), which for a non-canonical link
    # such as probit or cloglog differs from what drc and GenericLikelihoodModel report.
    #
    # There is no positive-definiteness gate on this intermediate,
    # centred-and-scaled-covariate-scale ("beta-scale") information matrix -- checking it would
    # be a bug: even a perfectly fine fit's beta-scale matrix can have a large or small
    # absolute magnitude for reasons that have nothing to do with whether the fit is degenerate
    # (for example a design with very few levels), and a positive-definiteness rule was never
    # meant to judge an intermediate quantity. Only a narrow inversion safety net remains here;
    # the real positive-definiteness decision is made once, below, on the natural-scale
    # covariance -- the only matrix a caller actually sees.
    observed_hessian = np.asarray(
        model.hessian(np.asarray(glm_result.params, dtype=np.float64), observed=True),
        dtype=np.float64,
    )
    neg_observed_information = -(observed_hessian + observed_hessian.T) / 2.0
    ok_warnings: tuple[str, ...] = ()
    if not bool(np.all(np.isfinite(neg_observed_information))):
        # statsmodels divides by zero at a cell whose fitted probability is exactly 0 or 1; the
        # information of the stable log-likelihood is finite there.
        _, numerical_hessian = _numerical_gradient_and_hessian(_beta_loglike, glm_beta)
        neg_observed_information = -(numerical_hessian + numerical_hessian.T) / 2.0
        if not _is_positive_definite(neg_observed_information):
            return _refit_by_direct_likelihood(
                obs,
                cells=cells,
                cluster_ids=cluster_ids,
                link=link,
                reason=_REFIT_INFORMATION_WARNING,
                mirrored=mirrored,
            )
        ok_warnings = (_INFORMATION_RECOVERED_WARNING,)
    try:
        sigma_beta = np.linalg.inv(neg_observed_information)
    except np.linalg.LinAlgError:
        return _status_only_fit(
            obs,
            cells=cells,
            cluster_ids=cluster_ids,
            link=link,
            status=Status.NOT_CONVERGED,
            fit_warnings=(
                "fit_dose_response: the GLM's observed information is singular -- no interior "
                "maximum was found",
            ),
        )
    if not bool(np.all(np.isfinite(sigma_beta))):
        return _status_only_fit(
            obs,
            cells=cells,
            cluster_ids=cluster_ids,
            link=link,
            status=Status.NOT_CONVERGED,
            fit_warnings=(
                "fit_dose_response: the GLM's observed information inverts to a non-finite "
                "covariance -- no interior maximum was found",
            ),
        )
    jacobian = spread * np.array(
        [[-1.0 / beta1, beta0 / beta1**2], [0.0, -1.0 / beta1**2]], dtype=np.float64
    )
    natural_cov = jacobian @ sigma_beta @ jacobian.T
    natural_cov = (natural_cov + natural_cov.T) / 2.0

    # Covariance's own positive-definiteness rule, on the correlation scale so the status does
    # not depend on severity units.
    if not _is_positive_definite(natural_cov):
        return _status_only_fit(
            obs,
            cells=cells,
            cluster_ids=cluster_ids,
            link=link,
            status=Status.NOT_CONVERGED,
            fit_warnings=(
                "fit_dose_response: the natural-scale covariance is not finite and "
                "positive-definite -- no interior maximum was found",
            ),
        )

    all_severity, all_successes, all_trials = _cells_to_arrays(cells)
    log_likelihood = _full_binomial_log_likelihood(
        all_severity,
        all_successes,
        all_trials,
        mu=mu,
        s=s,
        upper=1.0,
        lower=0.0,
        scale=axis.scale,
        link=link,
    )
    if not math.isfinite(log_likelihood):
        return _status_only_fit(
            obs,
            cells=cells,
            cluster_ids=cluster_ids,
            link=link,
            status=Status.NOT_CONVERGED,
            fit_warnings=(_NONFINITE_LOG_LIKELIHOOD_WARNING,),
        )

    params = {
        "mu": Parameter(value=mu, fixed=False),
        "s": Parameter(value=s, fixed=False),
        "upper": Parameter(value=1.0, fixed=True),
        "lower": Parameter(value=0.0, fixed=True),
    }
    covariance = Covariance(
        names=("mu", "s"), matrix=tuple(tuple(row) for row in natural_cov.tolist())
    )

    return Fit(
        axis=axis,
        outcome=obs.outcome,
        direction=obs.direction,
        model="binomial",
        link=link,
        params=params,
        covariance=covariance,
        log_likelihood=log_likelihood,
        cells=cells,
        status=Status.OK,
        cluster_ids=cluster_ids,
        warnings=ok_warnings,
        provenance={},
    )


# ================================================================================================
# The Gaussian (continuous-outcome) fit
# ================================================================================================
#
# The model is `P(x) = u - (u - l) * F(z)` with independent homoscedastic Gaussian errors, fitted by
# nonlinear least squares (`scipy.optimize.least_squares`, a library optimiser). The two asymptotes
# enter the mean linearly, so for a given `(mu, s)` they are the solution of a small weighted linear
# least-squares problem; that closed form (variable projection, Golub and Pereyra, 1973) ranks a
# deterministic grid of `(mu, log s)` and its best points start the optimiser.

# A fit whose asymptote gap `u - l` is below this fraction of the range of the level means has
# run to the boundary `u = l`, where the curve is constant.
_GAUSSIAN_GAP_FRACTION = 1.0e-3
# Relative slack in comparing residual sums of squares: a fit is "no better than" a bound, or "no
# worse than" another fit, when they differ by less than this fraction of the fit's own sum.
_GAUSSIAN_RSS_TOLERANCE = 1.0e-6
# The slack is also floored, in case the lack-of-fit part of the sum of squares is exactly zero or
# lost to rounding: this fraction of the larger of the within-level sum of squares and the
# weighted sum of squared deviations of the level means about their mean. Both are unchanged by
# adding a constant to every response, and both scale as the square of a multiplier, as the sum of
# squares does. A floor built on the raw level means (their sum of squares) would grow with an
# offset and refuse a fit that is well identified once a large constant is added to the data.
_GAUSSIAN_RSS_FLOOR = 1.0e-12
# The exponential-tail limits are profiled in the rate `lam * span`, in this step. The low end of
# the logarithm is a straight line. The high end puts the whole departure on the end level: it is
# the log of this multiple of the span over the gap between the end level and its neighbour (the
# departure is then below 1/40 of its peak one gap in), so a small end gap reaches far enough.
_GAUSSIAN_TAIL_LOG_RATE_LOW = -6.0
_GAUSSIAN_TAIL_END_FACTOR = 40.0
_GAUSSIAN_TAIL_STEP = 0.5
# Longest log-rate the tail profile evaluates, to keep exp() finite.
_GAUSSIAN_TAIL_LOG_RATE_CAP = 700.0
# Starts at an interior location, at these multiples of the median gap between tested levels.
_GAUSSIAN_INTERIOR_SCALES = (0.5, 1.0, 2.0)
# The `(mu, log s)` grid: `log s` runs from `_GAUSSIAN_LOG_S_BELOW` under to `_GAUSSIAN_LOG_S_ABOVE`
# over the log of the span of the tested levels, so a transition squeezed between two levels and a
# curve flat across the range are both covered. `mu` sits at each tested level, midway between
# adjacent levels, and beyond both ends by these multiples of the span (which also reaches a
# location between a log-axis control and the first nonzero level).
_GAUSSIAN_LOG_S_BELOW = math.log(1.0e8)
_GAUSSIAN_LOG_S_ABOVE = math.log(1.0e3)
_GAUSSIAN_LOG_S_STEP = 0.25
_GAUSSIAN_LOCATION_REACH = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
# How many of the grid's best local minima in `log s` start the optimiser, and how far from the
# data an asymptote may lie at a start, in ranges of the level means. A start with an asymptote a
# hundred ranges away is a grid point so flat that the asymptotes are fitted to noise in a
# near-straight line; the optimiser wanders there without converging, and spends its evaluations.
_GAUSSIAN_STARTS = 4
_GAUSSIAN_START_REACH = 100.0
# A 2x2 normal-equations determinant below this fraction of the product of its diagonal is a
# singular design for the asymptotes.
_GAUSSIAN_DETERMINANT_TOLERANCE = 1.0e-12
_GAUSSIAN_OPTIMIZER_TOLERANCE = 1.0e-12
_GAUSSIAN_MAX_EVALUATIONS = 150


def _f_pdf(z: NDArray[np.float64], link: str) -> NDArray[np.float64]:
    """The link's density ``F'(z)``."""
    if link == "probit":
        return np.asarray(norm.pdf(z), dtype=np.float64)
    if link == "logit":
        return np.asarray(expit(z) * expit(-z), dtype=np.float64)
    with np.errstate(over="ignore"):
        return np.asarray(np.exp(z - np.exp(z)), dtype=np.float64)  # cloglog


def _f_pdf_slope(z: NDArray[np.float64], link: str) -> NDArray[np.float64]:
    """The link's density slope ``F''(z)``."""
    if link == "probit":
        return np.asarray(-z * norm.pdf(z), dtype=np.float64)
    if link == "logit":
        return np.asarray(expit(z) * expit(-z) * (1.0 - 2.0 * expit(z)), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        growth = np.exp(z)
        slope = np.exp(z - growth) * (1.0 - growth)  # cloglog
    # Past the overflow of exp(z) the density has decayed to 0 faster than (1 - exp(z)) grows.
    return np.asarray(np.where(np.isfinite(growth), slope, 0.0), dtype=np.float64)


@dataclass(frozen=True)
class _GaussianProblem:
    """The pooled cells of a Gaussian series prepared for fitting: weights ``n``, means, the
    within-level sum of squares (which no curve changes), and the transformed severity of each
    cell centred and scaled for the optimiser (``t' = (t - centre) / spread``). A log-axis
    zero-severity control has no covariate value and is flagged in ``control``."""

    link: str
    weights: NDArray[np.float64]
    means: NDArray[np.float64]
    control: NDArray[np.bool_]
    t: NDArray[np.float64]
    ssw: float
    rows: int
    centre: float
    spread: float

    @property
    def levels(self) -> NDArray[np.float64]:
        """The transformed (centred, scaled) severities that carry a covariate value."""
        return np.sort(np.unique(self.t[~self.control]))


def _gaussian_problem(
    cells: tuple[ContinuousCell, ...],
    *,
    axis: Axis,
    link: str,
    shift: float = 0.0,
    scale: float = 1.0,
) -> _GaussianProblem:
    """Prepare ``cells`` for fitting. Responses are expressed as ``(y - shift) / scale`` (the
    identity by default), so that a fit is the same up to the map whatever the offset and unit of
    the data."""
    severity = np.array([c.severity for c in cells], dtype=np.float64)
    control = (severity == 0.0) if axis.scale == "log" else np.zeros(severity.size, dtype=bool)
    centre, spread = _centering_and_scale(severity, scale=axis.scale)
    t = np.zeros(severity.size, dtype=np.float64)
    if np.any(~control):
        t[~control] = (_transform(severity[~control], scale=axis.scale) - centre) / spread
    return _GaussianProblem(
        link=link,
        weights=np.array([float(c.n) for c in cells], dtype=np.float64),
        means=np.array([(c.mean - shift) / scale for c in cells], dtype=np.float64),
        control=control,
        t=t,
        ssw=math.fsum(c.sd * c.sd * (c.n - 1) for c in cells) / (scale * scale),
        rows=sum(c.n for c in cells),
        centre=centre,
        spread=spread,
    )


def _gaussian_cdf_parts(
    problem: _GaussianProblem, mu: NDArray[np.float64] | float, s: NDArray[np.float64] | float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``(z, F(z), 1 - F(z))`` for every cell, the latter two computed separately so each stays
    accurate in its own tail. ``mu`` and ``s`` broadcast against the cells on the last axis. The
    control has ``F = 0`` and ``1 - F = 1`` exactly, never through ``log(0)``."""
    with np.errstate(all="ignore"):
        z = (problem.t - mu) / s
        c = _f_cdf(z, problem.link)
        cb = _f_survival(z, problem.link)
    control = problem.control
    return (
        np.where(control, 0.0, z),
        np.where(control, 0.0, c),
        np.where(control, 1.0, cb),
    )


def _gaussian_projected_asymptotes(
    problem: _GaussianProblem,
    c: NDArray[np.float64],
    cb: NDArray[np.float64],
    *,
    fixed_upper: float | None,
    fixed_lower: float | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """The least-squares asymptotes ``(u, l)`` and the cell-level sum of squares
    ``sum n (ybar - u (1 - F) - l F)^2`` for each curve shape in ``c``/``cb`` (``F`` and ``1 - F``,
    shape ``(..., cells)``), the free asymptotes solved in closed form and the fixed ones held.

    A shape whose solution is singular or has ``u <= l`` (the family needs ``u > l``) gets an
    infinite sum of squares: a constant curve is the boundary of the family, not a member.
    """
    w, y = problem.weights, problem.means
    shape = c.shape[:-1]
    a11 = np.einsum("k,...k->...", w, cb * cb)
    a12 = np.einsum("k,...k->...", w, cb * c)
    a22 = np.einsum("k,...k->...", w, c * c)
    b1 = np.einsum("k,...k->...", w * y, cb)
    b2 = np.einsum("k,...k->...", w * y, c)
    with np.errstate(all="ignore"):
        if fixed_upper is None and fixed_lower is None:
            det = a11 * a22 - a12 * a12
            regular = det > _GAUSSIAN_DETERMINANT_TOLERANCE * a11 * a22
            upper = (b1 * a22 - b2 * a12) / det
            lower = (a11 * b2 - a12 * b1) / det
        elif fixed_lower is None:
            assert fixed_upper is not None
            regular = a22 > 0.0
            upper = np.full(shape, fixed_upper)
            lower = (b2 - fixed_upper * a12) / a22
        elif fixed_upper is None:
            regular = a11 > 0.0
            lower = np.full(shape, fixed_lower)
            upper = (b1 - fixed_lower * a12) / a11
        else:
            regular = np.ones(shape, dtype=bool)
            upper = np.full(shape, fixed_upper)
            lower = np.full(shape, fixed_lower)
        fitted = upper[..., None] * cb + lower[..., None] * c
        cost = np.einsum("k,...k->...", w, (y - fitted) ** 2)
    valid = regular & np.isfinite(upper) & np.isfinite(lower) & (upper > lower) & np.isfinite(cost)
    return upper, lower, np.where(valid, cost, np.inf)


def _gaussian_start_grid(
    problem: _GaussianProblem,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The ``mu`` and ``log s`` grids (primed units) that rank the optimiser's starts."""
    levels = problem.levels
    span = float(levels[-1] - levels[0]) if levels.size > 1 else 1.0
    if not span > 0.0:
        span = 1.0
    midpoints = 0.5 * (levels[1:] + levels[:-1])
    beyond = [
        end + sign * reach * span
        for reach in _GAUSSIAN_LOCATION_REACH
        for end, sign in ((levels[0], -1.0), (levels[-1], 1.0))
    ]
    mu = np.unique(np.concatenate([levels, midpoints, np.array(beyond)]))
    low = math.log(span) - _GAUSSIAN_LOG_S_BELOW
    high = math.log(span) + _GAUSSIAN_LOG_S_ABOVE
    count = math.floor((high - low) / _GAUSSIAN_LOG_S_STEP) + 1
    return mu, low + _GAUSSIAN_LOG_S_STEP * np.arange(count, dtype=np.float64)


def _gaussian_unpack(
    p: NDArray[np.float64], *, fixed_upper: float | None, fixed_lower: float | None
) -> tuple[float, float, float, float]:
    """``(mu, s, u, l)`` in the optimiser's units from its parameters: ``mu``, ``log s`` and, for
    each free asymptote, ``l`` and the log of the gap ``u - l`` (both free), or the log of the gap
    to the one fixed asymptote."""
    mu, s = float(p[0]), math.exp(float(p[1]))
    if fixed_upper is None and fixed_lower is None:
        lower = float(p[2])
        return mu, s, lower + math.exp(float(p[3])), lower
    if fixed_upper is not None and fixed_lower is None:
        return mu, s, fixed_upper, fixed_upper - math.exp(float(p[2]))
    if fixed_upper is None:
        assert fixed_lower is not None
        return mu, s, fixed_lower + math.exp(float(p[2])), fixed_lower
    assert fixed_lower is not None
    return mu, s, fixed_upper, fixed_lower


def _gaussian_residuals_and_jacobian(
    problem: _GaussianProblem, *, fixed_upper: float | None, fixed_lower: float | None
) -> tuple[
    Callable[[NDArray[np.float64]], NDArray[np.float64]],
    Callable[[NDArray[np.float64]], NDArray[np.float64]],
]:
    """The weighted residual vector ``sqrt(n) (ybar - fitted)`` and its analytic Jacobian in the
    optimiser's parameters (see :func:`_gaussian_unpack`)."""
    root_w = np.sqrt(problem.weights)

    def residuals(p: NDArray[np.float64]) -> NDArray[np.float64]:
        mu, s, upper, lower = _gaussian_unpack(p, fixed_upper=fixed_upper, fixed_lower=fixed_lower)
        _, c, cb = _gaussian_cdf_parts(problem, np.float64(mu), np.float64(s))
        with np.errstate(all="ignore"):
            return np.asarray(root_w * (problem.means - upper * cb - lower * c), dtype=np.float64)

    def jacobian(p: NDArray[np.float64]) -> NDArray[np.float64]:
        mu, s, upper, lower = _gaussian_unpack(p, fixed_upper=fixed_upper, fixed_lower=fixed_lower)
        z, c, cb = _gaussian_cdf_parts(problem, np.float64(mu), np.float64(s))
        with np.errstate(all="ignore"):
            gap = upper - lower
            density = np.where(problem.control, 0.0, _f_pdf(z, problem.link))
            columns = [gap * density / s, gap * density * z]
            if fixed_upper is None and fixed_lower is None:
                columns += [np.ones_like(c), gap * cb]
            elif fixed_lower is None:
                columns += [-gap * c]
            elif fixed_upper is None:
                columns += [gap * cb]
            return np.asarray(-root_w[:, None] * np.column_stack(columns), dtype=np.float64)

    return residuals, jacobian


def _gaussian_asymptote_start(
    upper: float, lower: float, *, fixed_upper: float | None, fixed_lower: float | None
) -> list[float]:
    """The optimiser's asymptote parameters (see :func:`_gaussian_unpack`) for ``(u, l)``."""
    gap = math.log(upper - lower)
    if fixed_upper is None and fixed_lower is None:
        return [lower, gap]
    if fixed_upper is None or fixed_lower is None:
        return [gap]
    return []


def _gaussian_step_rss(
    problem: _GaussianProblem, *, fixed_upper: float | None, fixed_lower: float | None
) -> float:
    """The smallest residual sum of squares of a step curve, the ``s -> 0`` limit of the family:
    the closed-form supremum that no fit with a finite scale can beat but may approach.

    The step sits between two adjacent tested levels (every such split is tried), or on one
    level, whose fitted value is then any value between the two asymptotes (as the scale shrinks
    the curve passes through it at any fraction). The zero-severity control of a log axis always
    lies on the upper side. For a split the asymptotes are solved in closed form, honouring fixed
    ones and the requirement ``u > l``. For a level on the step the middle value ``b`` is the
    argument of a convex one-dimensional problem, since the asymptotes are the group means
    clamped to ``b`` (or the fixed values). Infinite when no step qualifies.
    """
    levels = problem.levels
    t, control = problem.t, problem.control
    shapes = [
        np.where(control, 0.0, (t > 0.5 * (levels[k] + levels[k + 1])).astype(np.float64))
        for k in range(levels.size - 1)
    ]
    c = np.array(shapes, dtype=np.float64)
    _, _, cost = _gaussian_projected_asymptotes(
        problem, c, 1.0 - c, fixed_upper=fixed_upper, fixed_lower=fixed_lower
    )
    best = float(np.min(cost))
    w, y = problem.weights, problem.means

    def moments(mask: NDArray[np.bool_]) -> tuple[float, float, float]:
        weight = float(np.sum(w[mask]))
        if weight == 0.0:
            return 0.0, 0.0, 0.0
        mean = float(np.sum(w[mask] * y[mask])) / weight
        return weight, mean, float(np.sum(w[mask] * (y[mask] - mean) ** 2))

    for level in levels:
        on = ~control & (t == level)
        w_up, m_up, s_up = moments(control | (~control & (t < level)))
        w_mid, m_mid, s_mid = moments(on)
        w_low, m_low, s_low = moments(~control & (t > level))
        if w_up == 0.0 and w_low == 0.0:
            continue
        present = [m for weight, m in ((w_up, m_up), (w_mid, m_mid), (w_low, m_low)) if weight > 0]
        lowest = fixed_lower if fixed_lower is not None else min(present)
        highest = fixed_upper if fixed_upper is not None else max(present)
        if not lowest <= highest:
            continue

        def between(b: float) -> float:
            upper = fixed_upper if fixed_upper is not None else max(m_up, b)
            lower = fixed_lower if fixed_lower is not None else min(m_low, b)
            return float(
                w_up * (m_up - upper) ** 2 + w_mid * (m_mid - b) ** 2 + w_low * (m_low - lower) ** 2
            )

        run = minimize_scalar(
            between,
            bounds=(lowest, highest),
            method="bounded",
            options={"xatol": 1.0e-12 * (1.0 + abs(highest - lowest))},
        )
        value = min(
            float(run.fun),
            between(lowest),
            between(highest),
            between(m_mid) if lowest <= m_mid <= highest else math.inf,
        )
        best = min(best, s_up + s_mid + s_low + value)
    return problem.ssw + best


def _gaussian_tail_rss(
    problem: _GaussianProblem, *, fixed_upper: float | None, fixed_lower: float | None
) -> float:
    """The smallest residual sum of squares among the exponential-tail limits of the family, in
    which a free asymptote runs to infinity.

    Continuous responses are unbounded, so with a free asymptote the closure of the family holds
    curves ``a - A exp(lam t)`` (the lower asymptote to ``-infinity``; ``a`` is the upper
    asymptote, fixed or free) and ``a + A exp(-lam t)`` (the upper to ``+infinity``; ``a`` is the
    lower asymptote), with ``A >= 0`` and ``lam`` in ``[0, infinity]``, on the transformed
    covariate ``t``. The limit ``lam -> 0`` is a straight line (or a constant) and
    ``lam -> infinity`` puts the whole departure on the end level. For each ``lam`` the
    coefficients are weighted least squares under ``A >= 0``, in closed form, and ``lam`` is
    profiled on a log grid refined by a bounded scalar search.

    A log-axis zero-severity control has no covariate value. In the first family it sits at ``a``
    exactly (the term vanishes as ``t -> -infinity``). In the second it would sit at the upper
    asymptote, which is running to ``+infinity``, so that family is unavailable with a control.
    Infinite when neither family is allowed (both asymptotes fixed).
    """
    keep = ~problem.control
    levels = problem.levels
    w, y, t = problem.weights, problem.means, problem.t
    span = float(levels[-1] - levels[0])
    total = float(np.sum(w))
    mean_y = float(np.sum(w * y)) / total

    def fit_rss(basis: NDArray[np.float64], sign: float, anchor: float | None) -> float:
        d = sign * basis
        if anchor is not None:
            denom = float(np.sum(w * d * d))
            if not denom > 0.0:
                return math.inf
            amp = max(0.0, float(np.sum(w * d * (y - anchor))) / denom)
            return float(np.sum(w * (y - anchor - amp * d) ** 2))
        centred = d - float(np.sum(w * d)) / total
        denom = float(np.sum(w * centred * centred))
        if not denom > 0.0:
            return math.inf
        amp = max(0.0, float(np.sum(w * centred * (y - mean_y))) / denom)
        return float(np.sum(w * (y - mean_y - amp * centred) ** 2))

    def profile(orientation: str, anchor: float | None) -> float:
        def rss_at(log_x: float) -> float:
            lam = math.exp(log_x) / span
            with np.errstate(all="ignore"):
                if orientation == "upper":
                    basis = np.where(keep, np.exp(lam * (t - levels[-1])), 0.0)
                    return fit_rss(basis, -1.0, anchor)
                basis = np.where(keep, np.exp(-lam * (t - levels[0])), 0.0)
                return fit_rss(basis, 1.0, anchor)

        end_gap = (
            float(levels[-1] - levels[-2])
            if orientation == "upper"
            else float(levels[1] - levels[0])
        )
        high = min(
            math.log(_GAUSSIAN_TAIL_END_FACTOR * span / end_gap), _GAUSSIAN_TAIL_LOG_RATE_CAP
        )
        grid = np.arange(
            _GAUSSIAN_TAIL_LOG_RATE_LOW,
            max(high, _GAUSSIAN_TAIL_LOG_RATE_LOW + _GAUSSIAN_TAIL_STEP)
            + 0.5 * _GAUSSIAN_TAIL_STEP,
            _GAUSSIAN_TAIL_STEP,
        )
        values = np.array([rss_at(float(g)) for g in grid])
        if not np.any(np.isfinite(values)):
            return math.inf
        i = int(np.argmin(values))
        best = float(values[i])
        if 0 < i < grid.size - 1:
            run = minimize_scalar(
                rss_at, bounds=(float(grid[i - 1]), float(grid[i + 1])), method="bounded"
            )
            best = min(best, float(run.fun))
        return best

    best = math.inf
    if fixed_lower is None:
        best = min(best, profile("upper", fixed_upper))
    if fixed_upper is None and not np.any(problem.control):
        best = min(best, profile("lower", fixed_lower))
    # The profile above is over level means only; adding the within-level sum of squares puts the
    # supremum on the same total-RSS basis as the fit and the other suprema it is compared with.
    return problem.ssw + best


def _gaussian_flat_rss(
    problem: _GaussianProblem, *, fixed_upper: float | None, fixed_lower: float | None
) -> float | None:
    """The smallest residual sum of squares on the ``s -> infinity`` boundary, in closed form;
    ``None`` when both asymptotes are fixed (the boundary then has no closed form that a curve
    of the family approaches, and no flat check applies).

    With both asymptotes free the curve tends to a straight line in the covariate (the gap grows
    with the scale), so the supremum is the weighted linear regression of the level means on the
    covariate, with a slope that does not rise (a rising slope is the constant, the boundary of
    the family). With one asymptote fixed the curve tends to a constant that the fixed asymptote
    bounds: at most ``upper`` when the upper is fixed, at least ``lower`` when the lower is.
    A log-axis zero-severity control stays at ``upper`` however large the scale: at the fixed
    ``upper`` when that is fixed, and fitted exactly (its own free ``upper``) otherwise.
    """
    if fixed_upper is not None and fixed_lower is not None:
        return None
    keep = ~problem.control
    w, y, t = problem.weights[keep], problem.means[keep], problem.t[keep]
    total = float(np.sum(w))
    mean_y = float(np.sum(w * y)) / total
    if fixed_upper is None and fixed_lower is None:
        mean_t = float(np.sum(w * t)) / total
        centred = t - mean_t
        slope = float(np.sum(w * centred * (y - mean_y))) / float(np.sum(w * centred * centred))
        slope = min(slope, 0.0)
        residual = float(np.sum(w * (y - mean_y - slope * centred) ** 2))
    else:
        level = mean_y
        if fixed_upper is not None:
            level = min(level, fixed_upper)
        else:
            assert fixed_lower is not None
            level = max(level, fixed_lower)
        residual = float(np.sum(w * (y - level) ** 2))
        if fixed_upper is not None and np.any(problem.control):
            residual += float(
                np.sum(
                    problem.weights[problem.control]
                    * (problem.means[problem.control] - fixed_upper) ** 2
                )
            )
    return problem.ssw + residual


def _gaussian_observed_hessian(
    problem: _GaussianProblem,
    *,
    s: float,
    z: NDArray[np.float64],
    upper: float,
    lower: float,
    estimated: list[str],
) -> NDArray[np.float64]:
    """The observed Hessian ``1/2 d2 RSS / d theta^2`` on the natural parameters ``estimated``
    (a subset of ``mu, s, upper, lower``, in that order), computed analytically:

    ``sum_k n_k [ grad f_k grad f_k' - (ybar_k - f_k) hess f_k ]``

    for the cell means ``f = u - (u - l) F((x - mu) / s)``. It differs from the Gauss-Newton
    ``J'J`` by the residual-weighted curvature term, which does not vanish when the model is
    misspecified or the sample is small.
    """
    link = problem.link
    gap = upper - lower
    h = np.where(problem.control, 0.0, _f_pdf(z, link))
    hp = np.where(problem.control, 0.0, _f_pdf_slope(z, link))
    c = np.where(problem.control, 0.0, _f_cdf(z, link))
    cb = np.where(problem.control, 1.0, _f_survival(z, link))
    residual = problem.means - upper * cb - lower * c
    zero = np.zeros_like(z)
    gradient = {"mu": gap * h / s, "s": gap * h * z / s, "upper": cb, "lower": c}
    curvature = {
        ("mu", "mu"): -gap * hp / (s * s),
        ("mu", "s"): -gap * (z * hp + h) / (s * s),
        ("s", "s"): -gap * z * (z * hp + 2.0 * h) / (s * s),
        ("mu", "upper"): h / s,
        ("mu", "lower"): -h / s,
        ("s", "upper"): h * z / s,
        ("s", "lower"): -h * z / s,
        ("upper", "upper"): zero,
        ("upper", "lower"): zero,
        ("lower", "lower"): zero,
    }
    size = len(estimated)
    hessian = np.empty((size, size), dtype=np.float64)
    for i, a in enumerate(estimated):
        for j, b in enumerate(estimated):
            second = curvature[(a, b) if (a, b) in curvature else (b, a)]
            hessian[i, j] = float(
                np.sum(problem.weights * (gradient[a] * gradient[b] - residual * second))
            )
    return hessian


def _gaussian_log_likelihood(rss: float, rows: int) -> float:
    """The Gaussian log-likelihood at the maximum over the residual variance,
    ``-n/2 (log(2 pi rss / n) + 1)``."""
    return float(-0.5 * rows * (math.log(2.0 * math.pi * rss / rows) + 1.0))


def _gaussian_failure(
    reason: str,
    *,
    cells: tuple[ContinuousCell, ...],
    axis: Axis,
    outcome: str,
    direction: str,
    link: str,
    cluster_ids: tuple[str | int, ...] | None,
) -> Fit:
    """A ``NOT_CONVERGED`` Gaussian fit: the reason in ``warnings`` and no number anywhere."""
    return Fit(
        axis=axis,
        outcome=outcome,
        direction=direction,
        model="gaussian",
        link=link,
        params=None,
        covariance=None,
        log_likelihood=None,
        cells=cells,
        status=Status.NOT_CONVERGED,
        cluster_ids=cluster_ids,
        warnings=(f"fit_dose_response: {reason}",),
        provenance={},
    )


def _fit_gaussian_cells(
    cells: tuple[ContinuousCell, ...],
    *,
    axis: Axis,
    outcome: str,
    direction: str,
    cluster_ids: tuple[str | int, ...] | None,
    link: str,
    upper_is_estimate: bool,
    upper_value: float,
    lower_is_estimate: bool,
    lower_value: float,
) -> Fit:
    """Fit the decreasing Gaussian curve to pooled ``cells``. ``direction`` only words the
    reasons of a failure: an increasing outcome is negated before it gets here.

    The covariance is ``sigma^2 (1/2 d2 RSS / d theta^2)^-1`` at the optimum on the natural
    parameters, with the Hessian computed analytically and ``sigma^2 = RSS / df``.

    Every way the fit can fail to be identified is a named reason in ``warnings`` of a
    ``NOT_CONVERGED`` fit, with no parameter, covariance, likelihood or residual:

    - ``df``: no residual degrees of freedom (as many observations as parameters);
    - ``boundary``: the least-squares curve is constant (``u - l`` runs to 0), because the data
      do not fall with severity, or a fixed asymptote cannot be reached from the data;
    - ``step``, ``flat`` and ``unbounded asymptote``: the fit is no better than a supremum of the
      closure of the curve family, whatever its scale. Responses are unbounded, so a free
      asymptote can run to infinity. The suprema, all in closed form or a one-dimensional
      profile, are the ``s -> 0`` step (between adjacent tested levels, or on one level), the
      ``s -> infinity`` straight line or constant, and the exponential tails
      ``a - A exp(lam t)`` and ``a + A exp(-lam t)`` (``A >= 0``) that a free lower or upper
      asymptote running to infinity reaches. Both asymptotes fixed leaves only the step. The
      comparison allows a slack of a millionth of the lack-of-fit part of the sum of squares, plus
      a floor of 1e-12 times the larger of the within-level sum of squares and the weighted sum of
      squared deviations of the level means about their mean. The reason names the supremum the
      fit sits on. The same limits apply at the best point of the start grid when no start
      converged (the reason then says that no start converged and the grid point is no better than
      the limit, rather than reporting a proven non-identification). The fit is carried out on
      responses centred and scaled by their own spread and mapped back, so it does not depend on
      the offset or unit of the response; an overflow anywhere is a named ``optimizer`` failure;
    - ``singular``: the observed Hessian is not positive-definite;
    - ``optimizer``: the optimiser failed, produced a non-finite value, or ended above the best
      point of the start grid;
    - ``zero residual``: the curve passes through every observation, so there is no residual
      variance to calibrate with.
    """

    def failed(reason: str) -> Fit:
        return _gaussian_failure(
            reason,
            cells=cells,
            axis=axis,
            outcome=outcome,
            direction=direction,
            link=link,
            cluster_ids=cluster_ids,
        )

    # The fit is carried out on responses centred at their weighted mean and divided by their
    # weighted standard deviation, with any fixed asymptote moved the same way, and mapped back at
    # the end. The tolerances and floors below are then in units of the data's own spread, so the
    # result is the same under y -> c y + d: a large offset can neither overflow the optimiser's
    # exponentials nor push a relative floor above the lack of fit.
    rows_total = sum(c.n for c in cells)
    shift = math.fsum(c.n * c.mean for c in cells) / rows_total
    scale = math.sqrt(math.fsum(c.n * (c.mean - shift) ** 2 for c in cells) / rows_total)
    if not (math.isfinite(scale) and scale > 0.0):
        shift, scale = 0.0, 1.0
    fixed_upper = None if upper_is_estimate else (upper_value - shift) / scale
    fixed_lower = None if lower_is_estimate else (lower_value - shift) / scale
    problem = _gaussian_problem(cells, axis=axis, link=link, shift=shift, scale=scale)
    parameters = 2 + int(upper_is_estimate) + int(lower_is_estimate)
    df = problem.rows - parameters
    if df <= 0:
        return failed(
            f"the residual has no degrees of freedom (df={df}: {problem.rows} observation(s) for "
            f"{parameters} estimated parameter(s)); a Gaussian fit needs more observations than "
            "parameters"
        )
    if problem.levels.size < 2:
        return failed(
            "the information matrix is singular: fewer than two distinct severity levels carry a "
            "covariate value, so the location and scale of the transition are not identified"
        )

    if not float(np.ptp(problem.means)) > 0.0:
        return failed(
            "the least-squares curve runs to the boundary of the parameter space: every level "
            "has the same mean, so the best curve is constant (upper = lower)"
        )

    mu_grid, log_s_grid = _gaussian_start_grid(problem)
    _, c_grid, cb_grid = _gaussian_cdf_parts(
        problem, mu_grid[:, None, None], np.exp(log_s_grid)[None, :, None]
    )
    upper_grid, lower_grid, cost_grid = _gaussian_projected_asymptotes(
        problem, c_grid, cb_grid, fixed_upper=fixed_upper, fixed_lower=fixed_lower
    )
    if not np.any(np.isfinite(cost_grid)):
        return failed(
            "the least-squares curve runs to the boundary of the parameter space: no curve with "
            f"upper > lower follows the data for direction={direction!r}, so the response does "
            "not move the way the direction says, or a fixed asymptote cannot be reached from "
            "the data"
        )

    best_mu_index = np.argmin(cost_grid, axis=0)
    columns = np.arange(log_s_grid.size)
    profile = cost_grid[best_mu_index, columns]
    ranking = np.where(np.isfinite(profile), -profile, -np.inf)
    starts: list[NDArray[np.float64]] = []
    centre_of_means = float(np.mean(problem.means))
    reach = _GAUSSIAN_START_REACH * float(np.ptp(problem.means))
    for index in _top_local_maxima(ranking, _GAUSSIAN_STARTS):
        i = int(best_mu_index[index])
        upper_0, lower_0 = float(upper_grid[i, index]), float(lower_grid[i, index])
        if max(abs(upper_0 - centre_of_means), abs(lower_0 - centre_of_means)) > reach:
            continue
        head = [float(mu_grid[i]), float(log_s_grid[index])] + _gaussian_asymptote_start(
            upper_0, lower_0, fixed_upper=fixed_upper, fixed_lower=fixed_lower
        )
        starts.append(np.array(head, dtype=np.float64))

    # A start at an interior location, with a scale of the order of the gap between tested levels
    # and the asymptotes from the closed form, so that a fit is never judged from degenerate
    # starts alone.
    levels = problem.levels
    interior = mu_grid[(mu_grid >= levels[0]) & (mu_grid <= levels[-1])]
    typical_gap = float(np.median(np.diff(levels)))
    for factor in _GAUSSIAN_INTERIOR_SCALES:
        _, c_in, cb_in = _gaussian_cdf_parts(problem, interior[:, None], factor * typical_gap)
        upper_in, lower_in, cost_in = _gaussian_projected_asymptotes(
            problem, c_in, cb_in, fixed_upper=fixed_upper, fixed_lower=fixed_lower
        )
        if np.any(np.isfinite(cost_in)):
            j = int(np.argmin(cost_in))
            starts.append(
                np.array(
                    [
                        float(interior[j]),
                        math.log(factor * typical_gap),
                        *_gaussian_asymptote_start(
                            float(upper_in[j]),
                            float(lower_in[j]),
                            fixed_upper=fixed_upper,
                            fixed_lower=fixed_lower,
                        ),
                    ],
                    dtype=np.float64,
                )
            )

    residuals, jacobian = _gaussian_residuals_and_jacobian(
        problem, fixed_upper=fixed_upper, fixed_lower=fixed_lower
    )
    best: Any = None
    for start in starts:
        try:
            run = least_squares(
                residuals,
                start,
                jac=jacobian,
                method="trf",
                xtol=_GAUSSIAN_OPTIMIZER_TOLERANCE,
                ftol=_GAUSSIAN_OPTIMIZER_TOLERANCE,
                gtol=_GAUSSIAN_OPTIMIZER_TOLERANCE,
                max_nfev=_GAUSSIAN_MAX_EVALUATIONS,
            )
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            continue
        if run.status > 0 and np.all(np.isfinite(run.x)) and math.isfinite(run.cost):
            if best is None or run.cost < best.cost:
                best = run
    grid_rss = problem.ssw + float(np.min(cost_grid))
    step_rss = _gaussian_step_rss(problem, fixed_upper=fixed_upper, fixed_lower=fixed_lower)
    flat_rss = _gaussian_flat_rss(problem, fixed_upper=fixed_upper, fixed_lower=fixed_lower)
    tail_rss = _gaussian_tail_rss(problem, fixed_upper=fixed_upper, fixed_lower=fixed_lower)
    magnitude = max(problem.ssw, float(np.sum(problem.weights * problem.means**2)))
    reasons = {
        "flat": (
            flat_rss,
            "the fitted curve is flat across the tested range: no curve with a finite scale fits "
            "better than the straight line (or constant) it tends to, so the location and scale "
            "are not identified",
        ),
        "step": (
            step_rss,
            "the fitted curve is a step between adjacent tested levels, or on one level: no "
            "curve with a finite scale fits better than the step, so its location and scale are "
            "not identified",
        ),
        "unbounded asymptote": (
            tail_rss,
            "an asymptote runs to infinity (unbounded asymptote): no curve with finite "
            "asymptotes fits better than the exponential tail a - A exp(lam t) it tends to, so "
            "the asymptote and the scale are not identified",
        ),
    }

    def unidentified(rss_fit: float) -> str | None:
        """The name of the supremum of the closure of the family that the fit is no better than:
        the step, the straight line or constant, or an exponential tail, whatever its scale.
        It is the largest one the fit does not beat, the one it sits on."""
        slack = _GAUSSIAN_RSS_TOLERANCE * max(rss_fit - problem.ssw, 0.0)
        slack += _GAUSSIAN_RSS_FLOOR * magnitude
        reached = [
            (bound, name)
            for name, (bound, _) in reasons.items()
            if bound is not None and math.isfinite(bound) and rss_fit >= bound - slack
        ]
        return max(reached, key=lambda pair: pair[0])[1] if reached else None

    def grid_failed(name: str, why: str) -> Fit:
        return failed(
            f"{why}; the best point of the start grid is no better than the {name} limit "
            "of the curve family, so no interior optimum was found"
        )

    if best is None:
        name = unidentified(grid_rss)
        if name is not None:
            return grid_failed(name, "no start converged")
        return failed(
            "the optimizer did not converge to a finite least-squares solution from any of its "
            f"{len(starts)} start(s)"
        )

    try:
        mu_p, s_p, upper, lower = _gaussian_unpack(
            best.x, fixed_upper=fixed_upper, fixed_lower=fixed_lower
        )
    except OverflowError:
        return failed("the optimizer returned a non-finite value (overflow)")
    rss = problem.ssw + 2.0 * float(best.cost)
    if not all(math.isfinite(v) for v in (mu_p, s_p, upper, lower, rss)):
        return failed("the optimizer returned a non-finite value")
    if rss > grid_rss + _GAUSSIAN_RSS_TOLERANCE * grid_rss:
        name = unidentified(grid_rss)
        if name is not None:
            return grid_failed(name, "the optimizer ended above the start grid")
        return failed(
            f"the optimizer did not converge: its residual sum of squares ({rss * scale**2!r}) is "
            f"above the best point of the start grid ({grid_rss * scale**2!r})"
        )

    spread_of_means = float(np.ptp(problem.means))
    if not upper - lower > _GAUSSIAN_GAP_FRACTION * spread_of_means:
        return failed(
            "the least-squares curve runs to the boundary of the parameter space: the fitted "
            f"upper asymptote ({upper * scale + shift!r}) is not above the lower "
            f"({lower * scale + shift!r}) by more than "
            f"{_GAUSSIAN_GAP_FRACTION} of the range of the level means, so the curve is "
            f"constant; the response does not move the way direction={direction!r} says, or a "
            "fixed asymptote cannot be reached from the data"
        )
    name = unidentified(rss)
    if name is not None:
        return failed(reasons[name][1])
    if not rss > 0.0:
        return failed(
            "zero residual: the curve passes through every observation, so there is no residual "
            "variance to calibrate an interval with"
        )

    z_hat, _, _ = _gaussian_cdf_parts(problem, np.float64(mu_p), np.float64(s_p))
    s_natural = s_p * problem.spread
    mu_natural = problem.centre + problem.spread * mu_p
    estimated = ["mu", "s"]
    if upper_is_estimate:
        estimated.append("upper")
    if lower_is_estimate:
        estimated.append("lower")
    with np.errstate(all="ignore"):
        hessian = _gaussian_observed_hessian(
            problem, s=s_natural, z=z_hat, upper=upper, lower=lower, estimated=estimated
        )
    if not np.all(np.isfinite(hessian)) or not _is_positive_definite(0.5 * (hessian + hessian.T)):
        return failed(
            "the observed Hessian of the residual sum of squares is not positive-definite at "
            "the least-squares solution, so the parameters are not identified (singular)"
        )
    try:
        covariance_matrix = (rss / df) * np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        return failed("the observed Hessian is singular and cannot be inverted")
    # Back to the units of the data: the asymptotes carry the response's multiplier, the
    # location and scale of the transition do not.
    units = np.array([scale if name in ("upper", "lower") else 1.0 for name in estimated])
    covariance_matrix = covariance_matrix * np.outer(units, units)
    covariance_matrix = 0.5 * (covariance_matrix + covariance_matrix.T)
    if not np.all(np.isfinite(covariance_matrix)) or not _is_positive_definite(covariance_matrix):
        return failed(
            "the covariance matrix is not positive-definite at the least-squares solution, so "
            "the parameters are not identified"
        )
    params = {
        "mu": Parameter(value=mu_natural, fixed=False),
        "s": Parameter(value=s_natural, fixed=False),
        "upper": Parameter(
            value=upper * scale + shift if upper_is_estimate else upper_value,
            fixed=not upper_is_estimate,
        ),
        "lower": Parameter(
            value=lower * scale + shift if lower_is_estimate else lower_value,
            fixed=not lower_is_estimate,
        ),
    }
    rss *= scale * scale
    return Fit(
        axis=axis,
        outcome=outcome,
        direction=direction,
        model="gaussian",
        link=link,
        params=params,
        covariance=Covariance(
            names=tuple(estimated),
            matrix=tuple(tuple(float(v) for v in r) for r in covariance_matrix),
        ),
        log_likelihood=_gaussian_log_likelihood(rss, problem.rows),
        cells=cells,
        status=Status.OK,
        cluster_ids=cluster_ids,
        warnings=(),
        provenance={},
        residual=Residual(sigma=math.sqrt(rss / df), df=df, rss=rss),
    )


# ================================================================================================
# fit_dose_response: shared checks and the public entry point
# ================================================================================================


def _resolve_asymptote(value: float | str, *, label: str) -> tuple[bool, float]:
    """Validate ``upper``/``lower`` and split it into ``(is_estimate, fixed_value)``.
    ``fixed_value`` is meaningless (``math.nan``) when ``is_estimate`` is ``True``.
    """
    if isinstance(value, str):
        if value != _VALID_UPPER_LOWER:
            raise ValueError(
                f"fit_dose_response: {label} must be a float in [0, 1] or the string "
                f"'estimate', got {value!r}"
            )
        return True, math.nan
    _check_finite_number(value, label=f"fit_dose_response.{label}")
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"fit_dose_response: {label} must lie in [0, 1], got {value!r}")
    return False, float(value)


def _unique_cluster_ids(
    cluster: tuple[str | int, ...] | None,
) -> tuple[str | int, ...] | None:
    """The distinct cluster ids ``Observations`` carries, one instance per cluster (not one per
    row -- :class:`Fit`'s own invariant), in first-appearance order.
    """
    if cluster is None:
        return None
    return tuple(dict.fromkeys(cluster))


def _control_cell(cells: tuple[Cell, ...]) -> Cell | None:
    for cell in cells:
        if cell.severity == 0.0:
            return cell
    return None


def _check_control_incompatible(
    cells: tuple[Cell, ...],
    *,
    axis: Axis,
    upper_is_estimate: bool,
    upper_value: float,
    mirrored: bool = False,
) -> str | None:
    """A fixed ``upper=1.0`` with any failure at the zero-severity control
    drives the likelihood to zero on a log axis. Only meaningful on a log axis (a linear axis
    has no zero-severity control concept) and only when ``upper`` is fixed at exactly ``1.0``
    (a fixed ``upper`` at any other value does not create a zero-likelihood problem).

    With ``mirrored`` the arguments describe the complement of an increasing outcome, where the
    same condition reads as a lower asymptote fixed at ``0.0`` over a control with events, and
    the message says so.
    """
    if axis.scale != "log" or upper_is_estimate or upper_value != 1.0:
        return None
    control = _control_cell(cells)
    if control is None or control.successes == control.trials:
        return None
    if mirrored:
        return (
            "fit_dose_response: the zero-severity control has events but lower=0.0 is fixed, "
            'which drives the likelihood to zero on a log axis; refit with lower="estimate" '
            "instead"
        )
    return (
        "fit_dose_response: the zero-severity control has failures but upper=1.0 is fixed, "
        'which drives the likelihood to zero on a log axis; refit with upper="estimate" '
        "instead"
    )


def _check_separation(cells: tuple[Cell, ...], *, axis: Axis, mirrored: bool = False) -> str | None:
    """The pre-fit separation check: checked on
    nonzero levels on a log axis (the zero-severity control is excluded, since it carries no
    covariate information), or on every level on a linear axis (there is no special
    zero-severity control there). Triggers when no failure exists, no success exists, or the
    highest severity with a success is at or below the lowest severity with a failure (complete
    or quasi-complete separation, the latter including an exact tie at one level).

    With ``mirrored`` the cells are the complement of an increasing outcome, and the message
    swaps "success" and "failure" so that it speaks of the counted outcome.
    """
    checked = tuple(c for c in cells if not (axis.scale == "log" and c.severity == 0.0))
    if not checked:
        return None
    success, failure = ("failure", "success") if mirrored else ("success", "failure")
    has_success = any(c.successes > 0 for c in checked)
    has_failure = any(c.successes < c.trials for c in checked)
    if not has_success:
        return f"fit_dose_response: no {success} was observed at any tested severity (separation)"
    if not has_failure:
        return f"fit_dose_response: no {failure} was observed at any tested severity (separation)"
    max_success_severity = max(c.severity for c in checked if c.successes > 0)
    min_failure_severity = min(c.severity for c in checked if c.successes < c.trials)
    if max_success_severity <= min_failure_severity:
        return (
            "fit_dose_response: complete or quasi-complete separation -- the highest severity "
            f"with a {success} is at or below the lowest severity with a {failure}"
        )
    return None


def _status_only_fit(
    obs: Observations,
    *,
    cells: tuple[Cell, ...],
    cluster_ids: tuple[str | int, ...] | None,
    link: str,
    status: Status,
    fit_warnings: tuple[str, ...],
) -> Fit:
    return Fit(
        axis=obs.axis,
        outcome=obs.outcome,
        direction=obs.direction,
        model="binomial",
        link=link,
        params=None,
        covariance=None,
        log_likelihood=None,
        cells=cells,
        status=status,
        cluster_ids=cluster_ids,
        warnings=fit_warnings,
        provenance={},
    )


def _finalize(fit: Fit, *, monotonicity: MonotonicityCheck) -> Fit:
    """Attach ``monotonicity`` (every return path, whatever the status) and ``diagnostics``
    (only when ``fit.status is Status.OK``) to a freshly built ``Fit``. The single place
    :func:`fit_dose_response` routes every one of its return statements through, so the
    diagnostics/monotonicity rule is stated once.
    """
    diagnostics = diagnose(fit) if fit.status is Status.OK else None
    return replace(fit, diagnostics=diagnostics, monotonicity=monotonicity)


def _resolve_gaussian_asymptote(value: float | str, *, label: str) -> tuple[bool, float]:
    """Validate a Gaussian ``upper``/``lower`` and split it into ``(is_estimate, fixed_value)``.
    A fixed value is any finite response value, not a bool."""
    if isinstance(value, str):
        if value != _VALID_UPPER_LOWER:
            raise ValueError(
                f"fit_dose_response: {label} must be a finite response value or the string "
                f"'estimate', got {value!r}"
            )
        return True, math.nan
    _check_finite_number(value, label=f"fit_dose_response.{label}")
    return False, float(value)


def _fit_gaussian(
    obs: ContinuousObservations,
    *,
    link: str,
    upper: float | str,
    lower: float | str,
    dependence: str | None,
) -> Fit:
    """:func:`fit_dose_response` for a Gaussian series. An increasing series is fitted as the
    negation of the decreasing one (asymptotes ``-lower`` and ``-upper``) and mapped back, so
    both directions share the decreasing fit exactly."""
    monotonicity = check_monotonicity(obs, dependence=dependence)
    upper_is_estimate, upper_value = _resolve_gaussian_asymptote(upper, label="upper")
    lower_is_estimate, lower_value = _resolve_gaussian_asymptote(lower, label="lower")
    if not upper_is_estimate and not lower_is_estimate and not upper_value > lower_value:
        raise ValueError(
            f"fit_dose_response: upper ({upper_value!r}) must exceed lower ({lower_value!r})"
        )
    if obs.axis.scale == "log" and not any(severity != 0.0 for severity in obs.severity):
        raise ValueError(
            "fit_dose_response: this observation has no severity level besides the "
            "zero-severity control; a dose-response curve needs at least one severity level "
            "besides the control"
        )
    cluster_ids = _unique_cluster_ids(obs.cluster)
    if obs.direction == "decreasing":
        fit = _fit_gaussian_cells(
            obs.cells(),
            axis=obs.axis,
            outcome=obs.outcome,
            direction="decreasing",
            cluster_ids=cluster_ids,
            link=link,
            upper_is_estimate=upper_is_estimate,
            upper_value=upper_value,
            lower_is_estimate=lower_is_estimate,
            lower_value=lower_value,
        )
        return _finalize(fit, monotonicity=monotonicity)
    mirrored = _mirror_observations(obs)
    complement = _fit_gaussian_cells(
        mirrored.cells(),
        axis=mirrored.axis,
        outcome=mirrored.outcome,
        direction="decreasing",
        cluster_ids=cluster_ids,
        link=link,
        upper_is_estimate=lower_is_estimate,
        upper_value=-lower_value,
        lower_is_estimate=upper_is_estimate,
        lower_value=-upper_value,
    )
    return _mirror_fit(_finalize(complement, monotonicity=monotonicity))


@overload
def fit_dose_response(
    obs: Observations,
    *,
    model: Literal["binomial"],
    link: Literal["probit", "logit", "cloglog"],
    upper: float | Literal["estimate"],
    lower: float | Literal["estimate"],
    dependence: str | None = None,
) -> Fit: ...
@overload
def fit_dose_response(
    obs: ContinuousObservations,
    *,
    model: Literal["gaussian"],
    link: Literal["probit", "logit", "cloglog"],
    upper: float | Literal["estimate"],
    lower: float | Literal["estimate"],
    dependence: str | None = None,
) -> Fit: ...
def fit_dose_response(
    obs: Observations | ContinuousObservations,
    *,
    model: Literal["binomial", "gaussian"],
    link: Literal["probit", "logit", "cloglog"],
    upper: float | Literal["estimate"],
    lower: float | Literal["estimate"],
    dependence: str | None = None,
) -> Fit:
    """Fit a binomial or Gaussian dose-response curve to ``obs``.

    ``model="binomial"`` takes :class:`~robustmargin.Observations` (counts); ``model="gaussian"``
    takes :class:`~robustmargin.ContinuousObservations` (real-valued responses). See "A Gaussian
    series" below for the second.

    **Two fitting paths (binomial fits).** statsmodels ``GLM`` when both ``upper`` and ``lower`` are
    fixed at exactly ``1.0``/``0.0``; ``statsmodels`` ``GenericLikelihoodModel`` -- given a
    likelihood, never a hand-rolled optimizer or IRLS loop -- for every other combination of fixed
    and estimated asymptotes. A Gaussian fit has a single path, described under "A Gaussian
    series" below: nonlinear least squares with ``scipy.optimize.least_squares``, no ``GLM``
    and no ``GenericLikelihoodModel``. Both binomial paths are checked for separation before
    fitting and for a non-interior maximum after fitting; either gives an explicit
    :class:`~robustmargin.Status` with no numbers, never a value that merely happens to be where
    the optimizer stopped. Data that are a step between
    adjacent tested levels, with an estimated asymptote or a fixed one other than ``1``/``0``,
    have no finite maximum (the log-likelihood rises as ``s -> 0``) and give
    ``Status.NOT_CONVERGED``. With such asymptotes the overlap check alone is necessary but not
    sufficient.

    **Outcomes that rise with severity.** For ``obs.direction == "increasing"`` the counted
    outcome rises with severity: ``upper`` is the rate the curve approaches as severity grows and
    ``lower`` the rate at zero severity (so the control's own rate, on a log axis, is ``lower``).
    The fit is that of the complementary outcome (``trials - successes``) with asymptotes
    ``1 - lower`` and ``1 - upper``, mapped back; the likelihood, the optimisers and every status
    rule are the decreasing ones, and every message names the asymptotes, counts and direction
    of the outcome as supplied. With ``lower`` fixed at ``0`` a control that has any events is
    ``CONTROL_INCOMPATIBLE`` (refit with ``lower="estimate"``).

    Before any of that, :func:`robustmargin.check_monotonicity` is run on ``obs`` directly, and
    its result is attached to the returned ``Fit`` as ``monotonicity`` on *every* return path
    below, whatever the resulting ``status`` -- the check needs only raw counts, so it is not
    hidden behind a fit that failed. ``diagnostics`` is filled (via :func:`robustmargin.diagnose`)
    only when the fit reaches :data:`~robustmargin.Status.OK`.

    **A Gaussian series.** The curve is the same, ``P(x) = u - (u - l) * F(z)``, with independent
    homoscedastic normal errors, fitted by nonlinear least squares from a deterministic grid of
    starts (the asymptotes enter linearly, so the grid is ranked by their closed-form solution:
    variable projection, Golub and Pereyra, 1973). ``link="logit"``, ``"probit"`` and
    ``"cloglog"`` are R ``drc``'s ``LL.4``, ``LN.4`` and ``W1.4``; the covariance is the residual
    variance ``rss / df`` times the inverse of the observed Hessian ``1/2 d2 RSS / d theta2`` on
    the natural parameters, computed analytically (cf. Ritz, Baty, Streibig and Gerhard, 2015,
    *PLOS ONE* 10(12):e0146021). Fixed asymptotes cost no degree of freedom
    and are any finite response values with ``upper > lower``. An increasing series is the
    negation of the decreasing fit of the negated response, which is exact. A fit that cannot be
    identified is ``Status.NOT_CONVERGED`` with no number and one of these reasons in
    ``warnings``: ``df`` (no residual degrees of freedom), ``boundary`` (the curve runs to
    ``upper = lower``, because the response does not move the way ``direction`` says), ``step``,
    ``flat`` or ``unbounded asymptote`` (the fit is no better than a closed-form limit of the
    curve family: a step, a straight line or constant, or an exponential tail with an asymptote at
    infinity), ``singular`` (the observed Hessian is not
    positive-definite), ``optimizer`` (no finite least-squares solution, or one above the best
    point of the start grid) or ``zero residual``. There is no separation, no exact bound and no
    other model to fall back to; ``monotonicity`` (a one-sided Welch test between adjacent
    levels) is attached whatever the status.

    Parameters
    ----------
    obs
        The severity/outcome data to fit. Aggregated and per-observation ``Observations`` that
        pool to the same cells give numerically identical fits: everything below is a
        deterministic function of ``obs.cells()``, not of row order or representation. The same
        holds for a ``ContinuousObservations`` built by ``from_observations`` or ``from_summaries``,
        up to rounding.
    model
        ``"binomial"`` for ``Observations`` and ``"gaussian"`` for ``ContinuousObservations``.
    link
        ``"probit"``, ``"logit"`` or ``"cloglog"``. Fixed by the caller before the data is
        seen: this function never compares links and picks one for you.
    upper, lower
        Each either a fixed value or the literal string ``"estimate"``, always on the scale of
        the outcome as given. For the binomial model a fixed value is a probability in
        ``[0, 1]``; for the Gaussian model it is any finite response value (not a ``bool``)
        with ``upper > lower``. For a decreasing outcome ``upper`` is the zero-severity end; for
        an increasing outcome it is ``lower``. Required, with no default (a wrong default
        would silently change what curve is fit).
    dependence
        ``None`` (the default) resolves to ``"independent"`` unless ``obs.cluster`` is set, in
        which case omitting it raises -- forwarded to :func:`robustmargin.check_monotonicity`
        as-is, so there is only one implementation of this guard. ``"independent"`` may always
        be passed explicitly.

    Returns
    -------
    Fit
        ``status`` is :data:`~robustmargin.Status.OK` with all four natural-scale parameters and
        a covariance over the estimated ones, or one of :data:`~robustmargin.Status.SEPARATION`,
        :data:`~robustmargin.Status.NOT_CONVERGED` or
        :data:`~robustmargin.Status.CONTROL_INCOMPATIBLE`
        with no numbers and a reason in ``warnings``. ``cluster_ids`` mirrors ``obs.cluster``
        whatever the outcome. The dependence guard is enforced here too, at fit time (via the
        ``check_monotonicity`` call above) -- :class:`~robustmargin.Ratio` and
        :func:`robustmargin.threshold` each still enforce it again for their own clustered-fit
        inputs. ``monotonicity`` is never ``None`` on a freshly built ``Fit``; ``diagnostics`` is
        ``None`` unless ``status`` is ``OK``.

    Raises
    ------
    ValueError
        If ``model``/``link`` is not one of the values above, if ``obs`` is not the type ``model``
        takes, if ``upper``/``lower`` is not a valid probability (binomial) or finite value
        (Gaussian) or ``"estimate"``,
        if both are fixed with ``upper <= lower``, if ``obs.direction`` is ``"increasing"`` and
        ``lower`` is a fixed value other than 0 so small that ``1 - lower`` equals 1 in double
        precision, or if ``obs.cluster`` is set and
        ``dependence`` is not ``"independent"`` (raised by :func:`robustmargin.check_monotonicity`).
    """
    if model not in ("binomial", "gaussian"):
        raise ValueError(
            f"fit_dose_response: model must be 'binomial' or 'gaussian', got {model!r}"
        )
    if link not in _STATSMODELS_LINKS:
        raise ValueError(
            f"fit_dose_response: link must be one of {sorted(_STATSMODELS_LINKS)}, got {link!r}"
        )
    if isinstance(obs, ContinuousObservations) != (model == "gaussian"):
        raise ValueError(
            f"fit_dose_response: model={model!r} takes "
            f"{'ContinuousObservations' if model == 'gaussian' else 'Observations'}, got "
            f"{type(obs).__name__}"
        )
    if isinstance(obs, ContinuousObservations):
        return _fit_gaussian(obs, link=link, upper=upper, lower=lower, dependence=dependence)

    # Computed from the raw Observations before a fitting path is even chosen, and attached to
    # every return path below. This call is the dependence guard too -- there is no separate
    # check here, so there is only one implementation.
    monotonicity = check_monotonicity(obs, dependence=dependence)

    upper_is_estimate, upper_value = _resolve_asymptote(upper, label="upper")
    lower_is_estimate, lower_value = _resolve_asymptote(lower, label="lower")
    if not upper_is_estimate and not lower_is_estimate and not upper_value > lower_value:
        raise ValueError(
            f"fit_dose_response: upper ({upper_value!r}) must exceed lower ({lower_value!r})"
        )
    if (
        obs.direction == "increasing"
        and not lower_is_estimate
        and lower_value != 0.0
        and 1.0 - lower_value == 1.0
    ):
        # An increasing curve is fit as its complement, whose upper asymptote is `1 - lower`.
        raise ValueError(
            f"fit_dose_response: a fixed lower of {lower_value!r} is too close to 0 for an "
            "increasing outcome: it cannot be told from 0 at double precision in the "
            "complementary parametrisation the curve is fit in. Use lower=0.0 or a larger value"
        )

    if obs.direction == "decreasing":
        fit = _fit_counts(
            obs,
            link=link,
            upper_is_estimate=upper_is_estimate,
            upper_value=upper_value,
            lower_is_estimate=lower_is_estimate,
            lower_value=lower_value,
            mirrored=False,
        )
        return _finalize(fit, monotonicity=monotonicity)

    # The complementary outcome is fit as a decreasing curve with the asymptotes exchanged and
    # complemented, then mapped back.
    complement_fit = _fit_counts(
        _mirror_observations(obs),
        link=link,
        upper_is_estimate=lower_is_estimate,
        upper_value=1.0 - lower_value,
        lower_is_estimate=upper_is_estimate,
        lower_value=1.0 - upper_value,
        mirrored=True,
    )
    fit = _mirror_fit(_finalize(complement_fit, monotonicity=monotonicity))
    if fit.params is None:
        return fit
    # A fixed asymptote is reported as the caller gave it, not as `1 - (1 - value)`.
    params = dict(fit.params)
    if not upper_is_estimate:
        params["upper"] = Parameter(value=upper_value, fixed=True)
    if not lower_is_estimate:
        params["lower"] = Parameter(value=lower_value, fixed=True)
    return replace(fit, params=params)


def _fit_counts(
    obs: Observations,
    *,
    link: str,
    upper_is_estimate: bool,
    upper_value: float,
    lower_is_estimate: bool,
    lower_value: float,
    mirrored: bool,
) -> Fit:
    """The pre-fit checks and the choice of fitting path, for a decreasing ``obs``. ``mirrored``
    says that ``obs`` is the complement of an increasing outcome and so only selects the
    wording of a failure's reason."""
    cells = obs.cells()
    cluster_ids = _unique_cluster_ids(obs.cluster)

    control_incompatible_warning = _check_control_incompatible(
        cells,
        axis=obs.axis,
        upper_is_estimate=upper_is_estimate,
        upper_value=upper_value,
        mirrored=mirrored,
    )
    if control_incompatible_warning is not None:
        return _status_only_fit(
            obs,
            cells=cells,
            cluster_ids=cluster_ids,
            link=link,
            status=Status.CONTROL_INCOMPATIBLE,
            fit_warnings=(control_incompatible_warning,),
        )

    # A log-axis observation whose only severity level is the zero-severity control (the "no
    # covariate value at all" case, with nothing *else* in the data either) has no
    # likelihood-contributing level at all. Left unchecked, this
    # reaches `_centering_and_scale`/`_transformed_span` (both fitting paths) with an empty
    # nonzero-severity array and fails with numpy's own unhelpful "zero-size array to reduction
    # operation maximum which has no identity" rather than a message this package wrote itself.
    # Checked once, here, before either fitting path is chosen -- after
    # `_check_control_incompatible` so a control with failures still reports
    # CONTROL_INCOMPATIBLE, and before `_check_separation` so a control plus exactly one other
    # level still reaches that check and reports SEPARATION as before.
    if obs.axis.scale == "log" and not any(c.severity != 0.0 for c in cells):
        raise ValueError(
            "fit_dose_response: this observation has no severity level besides the "
            "zero-severity control; a dose-response curve needs at least one severity level "
            "besides the control"
        )

    separation_warning = _check_separation(cells, axis=obs.axis, mirrored=mirrored)
    if separation_warning is not None:
        return _status_only_fit(
            obs,
            cells=cells,
            cluster_ids=cluster_ids,
            link=link,
            status=Status.SEPARATION,
            fit_warnings=(separation_warning,),
        )

    if (
        not upper_is_estimate
        and not lower_is_estimate
        and upper_value == 1.0
        and lower_value == 0.0
    ):
        return _fit_glm(obs, cells=cells, cluster_ids=cluster_ids, link=link, mirrored=mirrored)

    return _fit_generic(
        obs,
        cells=cells,
        cluster_ids=cluster_ids,
        link=link,
        upper_is_estimate=upper_is_estimate,
        upper_value=upper_value,
        lower_is_estimate=lower_is_estimate,
        lower_value=lower_value,
        mirrored=mirrored,
    )
