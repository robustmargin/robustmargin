"""The dose-response fit result, its fitting entry point, and prediction.

This module holds :class:`Fit`, the two small dataclasses it embeds (:class:`Parameter`,
:class:`Covariance`), :func:`fit_dose_response`, and :class:`Prediction` (:meth:`Fit.predict`'s
return type). Two fitting paths exist: statsmodels ``GLM`` when both asymptotes are fixed at
exactly ``1.0``/``0.0``, and ``statsmodels.base.model.GenericLikelihoodModel`` -- given a
likelihood function, never a hand-rolled optimizer or IRLS loop -- otherwise. A fit with no
interior maximum (wrong-sign slope, or an estimated asymptote pinned to a parameter-space
boundary) is reported as ``Status.NOT_CONVERGED`` with the reason in ``Fit.warnings``, never as
a number that happens to be where the optimizer stopped.

**Known limitation.** A near-step curve (``s -> 0``) with an estimated asymptote, or a fixed
one other than ``1``/``0``, is not detected: it can return ``Status.OK`` with Wald standard
errors that carry little information. With such asymptotes the pre-fit overlap check is
necessary but not sufficient for an interior maximum; it is exact only for ``u = 1, l = 0``.

**v1 is binomial-only.** ``model`` accepts only ``"binomial"``; a new model family (``"ll4"``,
``"isotonic"``) is a later addition that requires a new ``schema_version``, not a silent
extension of the current one. :data:`robustmargin.report.SCHEMA_VERSION` is the one version
:func:`robustmargin.report.from_dict` reads and the single source of that number.

:class:`Fit` also carries :attr:`~Fit.diagnostics` (model-only goodness of fit, dispersion and
link comparison, filled only when ``status`` is ``OK`` -- see :func:`robustmargin.diagnose`) and
:attr:`~Fit.monotonicity` (a data-only adjacent-level check, filled on *every* return path
regardless of ``status`` -- see :func:`robustmargin.check_monotonicity` in
:mod:`robustmargin.validation`). Both are reported and never acted on: neither changes ``link``,
``params``, or any other field of the ``Fit`` they are attached to.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, gammaln, xlogy
from scipy.special import logit as _logit
from scipy.stats import norm
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning
from statsmodels.tools.tools import add_constant

from robustmargin.types import Axis, Cell, JSONValue, Observations, Status, _check_finite_number
from robustmargin.validation import Diagnostics, MonotonicityCheck, check_monotonicity, diagnose

# `robustmargin.validation` only imports `Fit` from this module under `TYPE_CHECKING` (see that
# module's own docstring), so this is not a circular import at runtime: by the time
# `fit_dose_response` below actually calls `diagnose`/`check_monotonicity`, this module has
# already finished executing this very import statement.

# Narrow statsmodels imports above, never `statsmodels.api`: that module also imports graphics and
# tsa, and statsmodels 0.14.0's tsa fails to import against current pandas
# (`deprecate_kwarg() missing 'new_arg_name'`), which breaks the dependency-floor CI job.

__all__ = ["Covariance", "Fit", "Parameter", "Prediction", "fit_dose_response"]

_VALID_MODELS = frozenset({"binomial"})
_VALID_LINKS: dict[str, frozenset[str]] = {
    "binomial": frozenset({"probit", "logit", "cloglog"}),
}
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
# passed the exact check. The check is exact for fixed u = 1, l = 0 (Albert & Anderson, 1984); with
# estimated or other fixed asymptotes it is necessary but not sufficient, and a near-step curve
# there is not yet detected (see the module docstring).

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
_WRONG_SIGN_OR_FLAT_WARNING = (
    "fit_dose_response: the fitted curve has no interior maximum for direction='decreasing' "
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

    The fitted curve is ``P(x) = u - (u - l) * F(z)``, where ``z = (log(x) - mu) / s`` on
    a log axis or ``z = (x - mu) / s`` on a linear axis, ``s > 0``, and ``F`` is the link's CDF
    (the standard normal CDF for ``probit``, the logistic CDF for ``logit``, or
    ``1 - exp(-exp(z))`` for ``cloglog``, which models the probability of *failure* rising with
    severity and so is not symmetric the way ``probit``/``logit`` are). ``mu`` is on the
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
    :class:`Parameter`. ``covariance`` is on that same natural scale: when asymptotes are
    estimated, the optimiser's own covariance (which is on its reparametrised scale) is
    delta-transformed onto the natural scale before being stored here.

    ``status == Status.OK`` if and only if both ``params`` and ``log_likelihood`` are present;
    when ``status`` is anything else, ``params``, ``covariance`` and ``log_likelihood`` are all
    ``None`` -- no number is ever produced for a failed fit. ``Status.UNREACHABLE`` and
    ``Status.FAILS_AT_BASELINE`` describe a *threshold*, not a fit (they depend on a ``Definition``
    a ``Fit`` does not have), and are never a ``Fit.status``.

    Binomial curves are decreasing-only in this release: ``direction="increasing"`` raises.
    The curve ``P = u - (u - l) * F(z)`` with ``s > 0`` is decreasing by construction, and
    increasing-is-worse outcomes are not yet supported. ``Observations`` itself still accepts
    both directions.

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
        ``"decreasing"`` only, in this release (see above).
    model
        The model family. Only ``"binomial"`` exists in this release; a new family such as
        ``"ll4"`` or ``"isotonic"`` is a later addition requiring a further ``schema_version``
        bump beyond the current one (:data:`robustmargin.report.SCHEMA_VERSION`).
    link
        The link function: ``"probit"``, ``"logit"`` or ``"cloglog"`` for ``model="binomial"``.
        Required; the schema's ``link`` enum carries no ``null`` alternative, so this is a
        genuine ``str``, never ``None``.
    params
        A mapping with exactly the keys ``{"mu", "s", "upper", "lower"}`` when ``status`` is
        ``OK``, else ``None``. ``s.value`` must be positive; ``lower.value`` and
        ``upper.value`` must each lie in ``[0, 1]`` with ``lower.value < upper.value``.
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
        The per-severity-level pooled counts the fit was made from
        (:meth:`~robustmargin.Observations.cells`). Never empty, and in strictly increasing
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
        carries a message suggesting ``upper="estimate"``). Empty by default.
    schema_version
        The serialised-result schema version this object belongs to. Defaults to ``"1"``, the
        only version ``from_dict`` reads; any other version is rejected.
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
        require it), in which case it reads back as ``None``. The last field, after
        ``diagnostics``.
    """

    axis: Axis
    outcome: str
    direction: str
    model: str
    link: str
    params: Mapping[str, Parameter] | None
    covariance: Covariance | None
    log_likelihood: float | None
    cells: tuple[Cell, ...]
    status: Status
    cluster_ids: tuple[str | int, ...] | None = None
    warnings: tuple[str, ...] = ()
    schema_version: str = "1"
    provenance: Mapping[str, JSONValue] = field(default_factory=dict)
    diagnostics: Diagnostics | None = None
    monotonicity: MonotonicityCheck | None = None

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
        if self.model == "binomial" and self.direction != "decreasing":
            raise ValueError(
                "Fit: model='binomial' only supports direction='decreasing' in this release; "
                "an increasing-is-worse definition is planned for a later release"
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

        if not self.cells:
            raise ValueError("Fit.cells must not be empty")
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
        if self.model == "binomial":
            if set(params) != _BINOMIAL_PARAM_NAMES:
                raise ValueError(
                    f"Fit.params for model='binomial' must have exactly the keys "
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
            if not (0.0 <= lower < upper <= 1.0):
                raise ValueError(
                    "Fit.params must satisfy 0 <= lower.value < upper.value <= 1, got "
                    f"lower={lower!r}, upper={upper!r}"
                )

            if self.axis.scale == "log" and upper == 1.0 and params["upper"].fixed:
                for cell in self.cells:
                    if cell.severity == 0.0 and cell.successes < cell.trials:
                        raise ValueError(
                            "Fit: the zero-severity control has failures but 'upper' is fixed "
                            "at exactly 1.0, so the likelihood is zero for this fit on a log "
                            'axis; refit with upper="estimate"'
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

        On a log axis, severity ``0.0`` is always the zero-severity control: its estimate
        is ``upper`` exactly, never computed through ``log(0)``. It is not a special case for
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
    :func:`scipy.special.xlogy` gives a 0-successes or 0-failures cell a term of exactly ``0.0``
    without ever evaluating ``log(0)``.

    ``center``/``spread`` are forwarded to :func:`_curve_success_probability` -- see there.
    Defaulted to the identity, so this still computes the reported ``Fit.log_likelihood`` on
    natural-scale ``mu``/``s`` against raw severity when a caller omits them (both fitting
    paths' final report does); only the estimated-asymptote optimiser's own iterations, which
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
    p = _curve_success_probability(
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
    terms = lchoose + xlogy(successes, p) + xlogy(failures, 1.0 - p)
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
) -> str | None:
    """An estimated asymptote pinned to a parameter-space boundary,
    checked both on the optimizer's own logit-scale parameter (``a``/``b``, ``|.| > 15``) and
    on the natural-scale probability itself (within ``1e-6`` of 0 or 1). Either test alone can
    miss a boundary fit the other catches (a huge logit-scale value that still rounds to a
    natural-scale probability inside the epsilon, or vice versa via the numerics of a
    particular optimizer step), so both are checked.
    """
    idx = 2
    if estimate_lower:
        a_value = float(final_params[idx])
        idx += 1
        if abs(a_value) > _ASYMPTOTE_LOGIT_BOUND:
            return (
                "fit_dose_response: the estimated lower asymptote reached the parameter "
                f"boundary (logit-scale a={a_value!r}, |a| > {_ASYMPTOTE_LOGIT_BOUND}) -- no "
                "interior maximum exists"
            )
    if estimate_upper:
        b_value = float(final_params[idx])
        if abs(b_value) > _ASYMPTOTE_LOGIT_BOUND:
            return (
                "fit_dose_response: the estimated upper asymptote reached the parameter "
                f"boundary (logit-scale b={b_value!r}, |b| > {_ASYMPTOTE_LOGIT_BOUND}) -- no "
                "interior maximum exists"
            )
    if estimate_lower and estimate_upper and lower < _ASYMPTOTE_PROBABILITY_EPS:
        return (
            "fit_dose_response: the estimated lower asymptote reached the probability "
            f"boundary (lower={lower!r} < {_ASYMPTOTE_PROBABILITY_EPS}) -- no interior maximum "
            "exists"
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
        return (
            "fit_dose_response: the estimated lower asymptote reached the probability "
            f"boundary near the fixed upper asymptote (upper - lower = {upper - lower!r} < "
            f"{_ASYMPTOTE_PROBABILITY_EPS}) -- no interior maximum exists"
        )
    if estimate_upper and upper > 1.0 - _ASYMPTOTE_PROBABILITY_EPS:
        return (
            "fit_dose_response: the estimated upper asymptote reached the probability "
            f"boundary (upper={upper!r} > 1 - {_ASYMPTOTE_PROBABILITY_EPS}) -- no interior "
            "maximum exists"
        )
    return None


def _failure_warnings(reason: str, *, pilot_wrong_sign: bool) -> tuple[str, ...]:
    """The ``warnings`` tuple for a ``NOT_CONVERGED`` estimated-asymptote fit: ``reason`` alone,
    or ``reason`` plus the shared wrong-sign/flat reason when a pilot GLM slope on the (rescaled)
    data had the wrong sign for ``direction='decreasing'`` -- unless ``reason`` already *is* that
    shared reason, in which case it is not duplicated.
    """
    if pilot_wrong_sign and reason != _WRONG_SIGN_OR_FLAT_WARNING:
        return (reason, _WRONG_SIGN_OR_FLAT_WARNING)
    return (reason,)


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
) -> Fit:
    """The estimated-or-non-trivial-fixed-asymptote path: ``statsmodels``
    ``GenericLikelihoodModel`` given :class:`_GenericLogLikelihood`, fit with ``method="bfgs"``
    then a ``method="newton"`` polish -- both statsmodels optimizers,
    never a hand-rolled Newton or IRLS loop.
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

    # The BFGS stage is only ever a starting point for the Newton polish: it routinely reports its
    # own `converged=False` / ConvergenceWarning even when it has landed close enough for Newton to
    # finish the job, so only the *final* (Newton) result's own convergence signal decides the
    # status below -- checking a warning merged across both stages would treat that expected
    # intermediate state as a failure.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bfgs_result = model.fit(
            start_params=start_params, method="bfgs", disp=False, maxiter=2000, gtol=1e-10
        )
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = model.fit(
                    start_params=np.asarray(bfgs_result.params, dtype=np.float64),
                    method="newton",
                    disp=False,
                    maxiter=200,
                    tol=1e-10,
                )
        except np.linalg.LinAlgError:
            # A singular Hessian during the Newton polish is usually the symptom of an estimated
            # asymptote drifting to its boundary; name that cause when the BFGS
            # stage already shows it, so the reason does not depend on which platform's linear
            # algebra happens to fail first.
            bfgs_params = np.asarray(bfgs_result.params, dtype=np.float64)
            _, _, bfgs_lower, bfgs_upper = _unpack_optimizer_params(
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
            ) or (
                "fit_dose_response: the optimizer's Hessian was singular while fitting -- "
                "no interior maximum was found"
            )
            return _status_only_fit(
                obs,
                cells=cells,
                cluster_ids=cluster_ids,
                link=link,
                status=Status.NOT_CONVERGED,
                fit_warnings=_failure_warnings(singular_reason, pilot_wrong_sign=pilot_wrong_sign),
            )
    convergence_warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    converged_flag = bool(result.mle_retvals.get("converged", False))
    final_params = np.asarray(result.params, dtype=np.float64)

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

    # A flat curve up to rounding, checked before anything that would build a covariance from a
    # near-degenerate slope.
    span = _transformed_span(severity, scale=axis.scale)
    if s > _FLAT_SLOPE_SPAN_FACTOR * span:
        return _status_only_fit(
            obs,
            cells=cells,
            cluster_ids=cluster_ids,
            link=link,
            status=Status.NOT_CONVERGED,
            fit_warnings=_failure_warnings(
                _WRONG_SIGN_OR_FLAT_WARNING, pilot_wrong_sign=pilot_wrong_sign
            ),
        )

    boundary_message = _boundary_warning(
        final_params,
        estimate_lower=lower_is_estimate,
        estimate_upper=upper_is_estimate,
        lower=lower,
        upper=upper,
    )
    if boundary_message is not None:
        return _status_only_fit(
            obs,
            cells=cells,
            cluster_ids=cluster_ids,
            link=link,
            status=Status.NOT_CONVERGED,
            fit_warnings=_failure_warnings(boundary_message, pilot_wrong_sign=pilot_wrong_sign),
        )

    # Newton's own `converged` flag only bounds the last step, so the final log-likelihood must
    # not have regressed below the BFGS stage's (relative tolerance -- an absolute tolerance here
    # does not scale with the log-likelihood's own magnitude, which depends on the sample size,
    # not on anything a caller controls), and the score at the final parameters must be near
    # zero.
    bfgs_llf = float(bfgs_result.llf)
    llf_regressed = float(result.llf) < bfgs_llf - _LLF_REGRESSION_TOLERANCE * (1.0 + abs(bfgs_llf))
    score = np.asarray(model.score(final_params), dtype=np.float64)
    score_tolerance = _SCORE_TOLERANCE_FACTOR * (1.0 + abs(float(result.llf)))
    score_not_near_zero = bool(np.any(np.abs(score) > score_tolerance))

    not_converged_reason: str | None = None
    if not converged_flag or convergence_warned:
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
            fit_warnings=_failure_warnings(not_converged_reason, pilot_wrong_sign=pilot_wrong_sign),
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
            ),
        )

    # The reported log-likelihood is always the full binomial log-likelihood over every cell
    # (`Fit.cells`, including a control `fit_cells` dropped above) -- `fit_cells`/`severity` etc.
    # above are the optimizer's own reduced view, not what gets reported.
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

    params = {
        "mu": Parameter(value=mu, fixed=False),
        "s": Parameter(value=s, fixed=False),
        "lower": Parameter(value=lower, fixed=not lower_is_estimate),
        "upper": Parameter(value=upper, fixed=not upper_is_estimate),
    }
    covariance = Covariance(
        names=tuple(names), matrix=tuple(tuple(row) for row in natural_cov.tolist())
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
        warnings=(),
        provenance={},
    )


# ================================================================================================
# fit_dose_response: the fixed u=1, l=0 path (statsmodels GLM)
# ================================================================================================


def _fit_glm(
    obs: Observations,
    *,
    cells: tuple[Cell, ...],
    cluster_ids: tuple[str | int, ...] | None,
    link: str,
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

    if convergence_warned or not bool(glm_result.converged):
        return _status_only_fit(
            obs,
            cells=cells,
            cluster_ids=cluster_ids,
            link=link,
            status=Status.NOT_CONVERGED,
            fit_warnings=("fit_dose_response: the GLM optimizer did not converge",),
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
            fit_warnings=(_WRONG_SIGN_OR_FLAT_WARNING,),
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
            fit_warnings=(_WRONG_SIGN_OR_FLAT_WARNING,),
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
        warnings=(),
        provenance={},
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
    cells: tuple[Cell, ...], *, axis: Axis, upper_is_estimate: bool, upper_value: float
) -> str | None:
    """A fixed ``upper=1.0`` with any failure at the zero-severity control
    drives the likelihood to zero on a log axis. Only meaningful on a log axis (a linear axis
    has no zero-severity control concept) and only when ``upper`` is fixed at exactly ``1.0``
    (a fixed ``upper`` at any other value does not create a zero-likelihood problem).
    """
    if axis.scale != "log" or upper_is_estimate or upper_value != 1.0:
        return None
    control = _control_cell(cells)
    if control is None or control.successes == control.trials:
        return None
    return (
        "fit_dose_response: the zero-severity control has failures but upper=1.0 is fixed, "
        'which drives the likelihood to zero on a log axis; refit with upper="estimate" '
        "instead"
    )


def _check_separation(cells: tuple[Cell, ...], *, axis: Axis) -> str | None:
    """The pre-fit separation check: checked on
    nonzero levels on a log axis (the zero-severity control is excluded, since it carries no
    covariate information), or on every level on a linear axis (there is no special
    zero-severity control there). Triggers when no failure exists, no success exists, or the
    highest severity with a success is at or below the lowest severity with a failure (complete
    or quasi-complete separation, the latter including an exact tie at one level).
    """
    checked = tuple(c for c in cells if not (axis.scale == "log" and c.severity == 0.0))
    if not checked:
        return None
    has_success = any(c.successes > 0 for c in checked)
    has_failure = any(c.successes < c.trials for c in checked)
    if not has_success:
        return "fit_dose_response: no success was observed at any tested severity (separation)"
    if not has_failure:
        return "fit_dose_response: no failure was observed at any tested severity (separation)"
    max_success_severity = max(c.severity for c in checked if c.successes > 0)
    min_failure_severity = min(c.severity for c in checked if c.successes < c.trials)
    if max_success_severity <= min_failure_severity:
        return (
            "fit_dose_response: complete or quasi-complete separation -- the highest severity "
            "with a success is at or below the lowest severity with a failure"
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


def fit_dose_response(
    obs: Observations,
    *,
    model: Literal["binomial"],
    link: Literal["probit", "logit", "cloglog"],
    upper: float | Literal["estimate"],
    lower: float | Literal["estimate"],
    dependence: str | None = None,
) -> Fit:
    """Fit a binomial dose-response curve to ``obs``.

    Two fitting paths: statsmodels ``GLM`` when both ``upper`` and ``lower`` are fixed at exactly
    ``1.0``/``0.0``; ``statsmodels`` ``GenericLikelihoodModel`` -- given a likelihood, never a
    hand-rolled optimizer or IRLS loop -- for every other combination of fixed and estimated
    asymptotes. Both paths are checked for separation before fitting and for a non-interior maximum
    after fitting; either gives an explicit :class:`~robustmargin.Status` with no numbers, never a
    value that merely happens to be where the optimizer stopped. One case is not yet detected: a
    near-step curve (``s -> 0``) with an estimated asymptote, or a fixed one other than
    ``1``/``0``, can return ``Status.OK`` with Wald standard errors that carry little
    information. With such asymptotes the overlap check is necessary but not sufficient.

    Before any of that, :func:`robustmargin.check_monotonicity` is run on ``obs`` directly, and
    its result is attached to the returned ``Fit`` as ``monotonicity`` on *every* return path
    below, whatever the resulting ``status`` -- the check needs only raw counts, so it is not
    hidden behind a fit that failed. ``diagnostics`` is filled (via :func:`robustmargin.diagnose`)
    only when the fit reaches :data:`~robustmargin.Status.OK`.

    Parameters
    ----------
    obs
        The severity/outcome data to fit. Aggregated and per-observation ``Observations`` that
        pool to the same cells give numerically identical fits: everything below is a
        deterministic function of ``obs.cells()``, not of row order or representation.
    model
        Only ``"binomial"`` exists in this release.
    link
        ``"probit"``, ``"logit"`` or ``"cloglog"``. Fixed by the caller before the data is
        seen: this function never compares links and picks one for you.
    upper, lower
        Each either a fixed probability in ``[0, 1]``, or the literal string ``"estimate"``.
        Required, with no default (a wrong default would silently change what curve is fit).
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
        If ``model``/``link`` is not one of the values above, if ``obs.direction`` is not
        ``"decreasing"`` (an increasing-is-worse curve is not yet supported; ``Fit``
        itself rejects it), if ``upper``/``lower`` is not a valid probability or ``"estimate"``,
        if both are fixed with ``upper <= lower``, or if ``obs.cluster`` is set and
        ``dependence`` is not ``"independent"`` (raised by :func:`robustmargin.check_monotonicity`).
    """
    if model != "binomial":
        raise ValueError(f"fit_dose_response: model must be 'binomial', got {model!r}")
    if link not in _STATSMODELS_LINKS:
        raise ValueError(
            f"fit_dose_response: link must be one of {sorted(_STATSMODELS_LINKS)}, got {link!r}"
        )
    if obs.direction != "decreasing":
        raise ValueError(
            "fit_dose_response: direction='increasing' is not supported in this release "
            "(model='binomial' only supports 'decreasing'; an increasing-is-worse definition "
            "is planned for a later release)"
        )

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

    cells = obs.cells()
    cluster_ids = _unique_cluster_ids(obs.cluster)

    control_incompatible_warning = _check_control_incompatible(
        cells, axis=obs.axis, upper_is_estimate=upper_is_estimate, upper_value=upper_value
    )
    if control_incompatible_warning is not None:
        return _finalize(
            _status_only_fit(
                obs,
                cells=cells,
                cluster_ids=cluster_ids,
                link=link,
                status=Status.CONTROL_INCOMPATIBLE,
                fit_warnings=(control_incompatible_warning,),
            ),
            monotonicity=monotonicity,
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

    separation_warning = _check_separation(cells, axis=obs.axis)
    if separation_warning is not None:
        return _finalize(
            _status_only_fit(
                obs,
                cells=cells,
                cluster_ids=cluster_ids,
                link=link,
                status=Status.SEPARATION,
                fit_warnings=(separation_warning,),
            ),
            monotonicity=monotonicity,
        )

    if (
        not upper_is_estimate
        and not lower_is_estimate
        and upper_value == 1.0
        and lower_value == 0.0
    ):
        return _finalize(
            _fit_glm(obs, cells=cells, cluster_ids=cluster_ids, link=link),
            monotonicity=monotonicity,
        )

    return _finalize(
        _fit_generic(
            obs,
            cells=cells,
            cluster_ids=cluster_ids,
            link=link,
            upper_is_estimate=upper_is_estimate,
            upper_value=upper_value,
            lower_is_estimate=lower_is_estimate,
            lower_value=lower_value,
        ),
        monotonicity=monotonicity,
    )
