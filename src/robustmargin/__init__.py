"""Robustness-margin estimation: thresholds with confidence intervals.

The package exports the vocabulary types (:class:`Axis`, :class:`Observations`,
:class:`ContinuousObservations`, :class:`Definition`, :class:`Cell`, :class:`ContinuousCell`,
:class:`Residual`, :class:`BootstrapSummary`, and :class:`ClusterCell` and
:class:`ClusterAdjacentPair` for clustered counts), the status and interval-shape enums
(:class:`Status`, :class:`IntervalShape`, alongside :class:`Censoring`), the result dataclasses
(:class:`Fit`, :class:`Threshold`, :class:`Ratio`, :class:`Scorecard`), the grid break-point rule
(:func:`grid_break_point`, returning :class:`GridBreakPoint`), and the dose-response fitter
(:func:`fit_dose_response`, returning a :class:`Fit` whose :meth:`~Fit.predict` returns a
:class:`Prediction`; ``model="isotonic"`` fits a monotone, parameter-free :class:`Fit` whose
values are ``Fit.cell_estimates``, and has no ``predict``). The JSON report module
(``robustmargin.report``) and the schema-valid test fakes (``robustmargin.testing``) are
importable as submodules but are not re-exported here.

:func:`threshold` solves for the severity at which a fitted curve crosses a target performance and
attaches an interval or an explicit censoring label, together with :class:`BaselineRate`,
:class:`ExactRates` and :func:`per_cell_clopper_pearson`. Its interval is a profile-likelihood,
delta-method or bootstrap one (``"parametric_bootstrap"`` and ``"cell_bootstrap"``, with
``n_resamples`` and ``seed``; :class:`BootstrapSummary` records what the replicates did).

:func:`ratio_interval` solves for the ratio of two thresholds' severities with a Fieller or
log-delta confidence interval.

:class:`Diagnostics` and :func:`diagnose` give model-only goodness of fit, dispersion and link
comparison for a converged :class:`Fit` (:class:`ContinuousDiagnostics` for a Gaussian one).
:class:`MonotonicityCheck`, :class:`AdjacentPair`, :class:`ContinuousAdjacentPair` and
:func:`check_monotonicity` are a data-only adjacent-severity-level check that needs only raw
counts or responses, computed on every :func:`fit_dose_response` return path regardless of
``Status``. Both are reported and never acted on automatically: neither changes a fitted
parameter, a reported threshold, or which link a caller chose.

Note that ``robustmargin.threshold`` is the *function*, not the submodule.
"""

from __future__ import annotations

from robustmargin.empirical import (
    BaselineRate,
    ExactRates,
    GridBreakPoint,
    grid_break_point,
    per_cell_clopper_pearson,
)
from robustmargin.models import Covariance, Fit, Parameter, Prediction, fit_dose_response
from robustmargin.ratio import Ratio, ratio_interval
from robustmargin.report import Scorecard
from robustmargin.threshold import Threshold, threshold
from robustmargin.types import (
    Axis,
    BootstrapSummary,
    Cell,
    Censoring,
    ClusterAdjacentPair,
    ClusterCell,
    ContinuousCell,
    ContinuousObservations,
    Definition,
    IntervalShape,
    Observations,
    Residual,
    Status,
)
from robustmargin.validation import (
    AdjacentPair,
    ContinuousAdjacentPair,
    ContinuousDiagnostics,
    Diagnostics,
    MonotonicityCheck,
    check_monotonicity,
    diagnose,
)

__version__ = "0.3.0"

__all__ = [
    "AdjacentPair",
    "Axis",
    "BaselineRate",
    "BootstrapSummary",
    "Cell",
    "Censoring",
    "ClusterAdjacentPair",
    "ClusterCell",
    "ContinuousAdjacentPair",
    "ContinuousCell",
    "ContinuousDiagnostics",
    "ContinuousObservations",
    "Covariance",
    "Definition",
    "Diagnostics",
    "ExactRates",
    "Fit",
    "GridBreakPoint",
    "IntervalShape",
    "MonotonicityCheck",
    "Observations",
    "Parameter",
    "Prediction",
    "Ratio",
    "Residual",
    "Scorecard",
    "Status",
    "Threshold",
    "__version__",
    "check_monotonicity",
    "diagnose",
    "fit_dose_response",
    "grid_break_point",
    "per_cell_clopper_pearson",
    "ratio_interval",
    "threshold",
]
