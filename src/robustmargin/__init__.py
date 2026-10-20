"""Robustness-margin estimation: thresholds with confidence intervals.

The package exports the vocabulary types (:class:`Axis`, :class:`Observations`,
:class:`Definition`, :class:`Cell`), the status and interval-shape enums (:class:`Status`,
:class:`IntervalShape`, alongside :class:`Censoring`), the result dataclasses (:class:`Fit`,
:class:`Threshold`, :class:`Ratio`, :class:`Scorecard`), the grid break-point rule
(:func:`grid_break_point`, returning :class:`GridBreakPoint`), and the dose-response fitter
(:func:`fit_dose_response`, returning a :class:`Fit` whose :meth:`~Fit.predict` returns a
:class:`Prediction`). The JSON report module (``robustmargin.report``) and the schema-valid test
fakes (``robustmargin.testing``) are importable as submodules but are not re-exported here.

:func:`threshold` solves for the severity at which a fitted curve crosses a target performance and
attaches an interval or an explicit censoring label, together with :class:`BaselineRate`,
:class:`ExactRates` and :func:`per_cell_clopper_pearson`.

:func:`ratio_interval` solves for the ratio of two thresholds' severities with a Fieller or
log-delta confidence interval.

:class:`Diagnostics` and :func:`diagnose` give model-only goodness of fit, dispersion and link
comparison for a converged :class:`Fit`. :class:`MonotonicityCheck`, :class:`AdjacentPair` and
:func:`check_monotonicity` are a data-only adjacent-severity-level check that needs only raw
counts, computed on every :func:`fit_dose_response` return path regardless of ``Status``. Both are
reported and never acted on automatically: neither changes a fitted parameter, a reported
threshold, or which link a caller chose.

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
    Cell,
    Censoring,
    Definition,
    IntervalShape,
    Observations,
    Status,
)
from robustmargin.validation import (
    AdjacentPair,
    Diagnostics,
    MonotonicityCheck,
    check_monotonicity,
    diagnose,
)

__version__ = "0.2.0"

__all__ = [
    "AdjacentPair",
    "Axis",
    "BaselineRate",
    "Cell",
    "Censoring",
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
