"""Unit tests for ``robustmargin.validation``'s **model-only** diagnostics:
:class:`~robustmargin.Diagnostics` and :func:`~robustmargin.diagnose` (plan section 5.7,
``docs/decisions/0023-fit-diagnostics-and-schema-v3.md``, as amended by that decision's
"Amendment 1").

Amendment 1 moves the adjacent-pair monotonicity check out of ``Diagnostics`` entirely and onto
``Fit.monotonicity`` (a :class:`~robustmargin.MonotonicityCheck`, populated on *every*
``fit_dose_response`` return path, whatever the ``Status``) -- because a check that needs only
raw counts should not disappear when the curve fit fails. Those tests live in
``tests/unit/test_monotonicity.py`` now, not here. This file covers exactly what
``Diagnostics`` keeps: goodness of fit, dispersion, and link comparison -- all of which still
require a converged fit and are still ``None`` for a non-``OK`` one.

Written before the implementation exists (plan section 6, ``CLAUDE.md``'s working rhythm):
every expected number here is computed *independently* of ``robustmargin.validation`` -- directly
from the binomial deviance/Pearson/AIC formulas via ``scipy``, or by a from-scratch Holm
step-down -- exactly the "unit test that computes the formulas directly" decisions/0023 itself
promises for the estimated-asymptote path (which has no R ``glm`` equivalent). ``diagnose()`` is
expected to agree with that independent computation to floating-point precision, not to within a
loose tolerance: a formula bug should never hide behind a wide ``pytest.approx``.

**Which cells count towards goodness of fit** (``docs/decisions/0023-fit-diagnostics-and-schema-
v3.md``'s "Amendment 2"; this file's own C1 finding). A cell is excluded from deviance, Pearson
chi-square and the cell count feeding ``df`` **only** when it is *structurally* deterministic
under the model: a log-axis cell at severity exactly ``0.0`` when ``upper`` is *fixed* at ``1.0``
(``direction="decreasing"``). That cell has zero variance and zero deviance by construction --
excluding it is what makes the diagnostics invariant to an all-success control. Every other cell
is counted, including one whose *computed* fitted probability happens to round to ``0`` or ``1``
in double precision (probabilities are clamped to ``[eps, 1 - eps]``, ``eps`` = R's
``.Machine$double.eps``, exactly as R's own ``binomial()$linkinv`` does, so such a cell
contributes approximately zero to both statistics and exactly ``1`` to ``df`` rather than a
``0/0``). Under an *estimated* ``upper``, the control identifies ``u`` and is always counted,
even if its value happens to equal ``1.0`` -- ``TestControlCountedWhenUpperIsEstimated`` below
pins this down directly.

**This corrects an earlier, wrong version of this same paragraph** (excluding by *computed*
value rather than by structure), which is exactly the bug Amendment 2's own "C1" finding
describes: excluding whenever a computed probability rounded to 0 or 1 made ``df`` depend on the
link (7 for probit/cloglog, 8 for logit, on the saturating grid ``TestStructuralDfExclusion``
below pins to 8 for all three), because probit/logit/cloglog saturate to floating-point 0/1 at
different ``z`` magnitudes. R's own ``glm`` never has this problem, and R's convention that
motivated the original (wrong) rule was actually the *structural* one all along -- see the
factual correction in ``tests/reference/test_diagnostics_drc.py``'s own module docstring.

**``SPARSE_CELLS:`` (Amendment 2 item 3, corrected by Amendment 3).** Amendment 2's own wording
flagged a counted cell whenever the expected count on a side (successes or failures) fell below
1, with no further condition. Amendment 3 found that this fires on essentially every correctly
specified rich-design fit -- an all-success or all-failure *plateau* cell far from the transition
has an expected count near zero on its "sparse" side, but since nothing was ever actually
observed there, that near-zero expectation is not a finding, just an artifact of clean data. The
corrected rule flags a side only when **both** the expected count is below a cutoff **and** the
observed count on that same side is at least 1 -- a genuine event against a small expectation,
which is the actual pathology the warning exists for (zeta PID x sensor_noise's 1 observed
failure where about 0.005 were expected).

**Amendment 4 tightened the cutoff from below 1 to below 0.1.** Amendment 3 alone still fired on
about 57.5% of correctly specified fits on the rich design (control + ``logspace(-2, 2, 10)``,
n=400, probit, ``mu=0``, ``s=0.5``): two grid points near the transition (about 0.215 and 4.64)
expect about 0.43 events on their sparse side, so a single observed event there is ordinary, not
a finding -- this is exactly the near-transition risk this file's own
``TestSparseCellsSilentOnCleanPlateaus`` documented empirically (11 of 20 seeded replicates fired
under Amendment 3) before Amendment 4 was decided. At the tightened cutoff those two cells no
longer qualify at all (0.43 is not below 0.1), which is what makes the rich design's false-warning
rate drop to about 0%. ``TestSparseCellsWarning`` below now needs three roles, not two, to pin
both conditions: a cell below the cutoff with an event (flagged), a cell below the cutoff with no
event (silent, Amendment 3's own case), and a cell *at or above* the new cutoff but still below
the old one, with an event (silent, pinning the 0.1 boundary itself).
"""

from __future__ import annotations

import dataclasses
import math
import re

import numpy as np
import pytest
from scipy.special import gammaln, xlogy
from scipy.stats import norm

from robustmargin import (
    Axis,
    Cell,
    Covariance,
    Diagnostics,
    Fit,
    Observations,
    Parameter,
    Status,
    diagnose,
    fit_dose_response,
)

_LOG_AXIS = Axis(name="severity", unit="unit", scale="log")
_LINEAR_AXIS = Axis(name="severity", unit="unit", scale="linear")


def _loglik(cells: tuple[Cell, ...], p: list[float]) -> float:
    """The full binomial log-likelihood at known per-cell success probabilities -- the same
    ``lchoose``-inclusive formula ``models.py`` uses, recomputed here from scratch so this file
    never imports robustmargin's own private helpers."""
    total = 0.0
    for cell, p_i in zip(cells, p, strict=True):
        y = float(cell.successes)
        n = float(cell.trials)
        lchoose = float(gammaln(n + 1.0) - gammaln(y + 1.0) - gammaln(n - y + 1.0))
        total += lchoose + float(xlogy(y, p_i)) + float(xlogy(n - y, 1.0 - p_i))
    return total


def _direct_gof(cells: tuple[Cell, ...], p: list[float], *, k: int) -> tuple[float, float, int]:
    """Deviance, Pearson chi-square and df computed directly from known fitted probabilities,
    independent of ``diagnose()``. No cell here has ``p_i`` at exactly 0 or 1 (that boundary case
    is exercised separately below), so every cell counts towards both sums and ``df``."""
    deviance = 0.0
    pearson = 0.0
    for cell, p_i in zip(cells, p, strict=True):
        y = float(cell.successes)
        n = float(cell.trials)
        mu = n * p_i
        deviance += 2.0 * (float(xlogy(y, y / mu)) + float(xlogy(n - y, (n - y) / (n - mu))))
        pearson += (y - mu) ** 2 / (n * p_i * (1.0 - p_i))
    return deviance, pearson, len(cells) - k


def _probit_p(mu: float, s: float, upper: float, lower: float, severity: float) -> float:
    """Success probability on a **log** axis: ``z = (log(severity) - mu) / s``."""
    z = (math.log(severity) - mu) / s
    return float(upper - (upper - lower) * norm.cdf(z))


def _probit_p_linear(mu: float, s: float, upper: float, lower: float, severity: float) -> float:
    """Success probability on a **linear** axis: ``z = (severity - mu) / s`` -- no ``log``, so
    ``severity == 0.0`` is an ordinary point, not R3's log-axis zero-severity control."""
    z = (severity - mu) / s
    return float(upper - (upper - lower) * norm.cdf(z))


def _fixed_fit(
    *,
    axis: Axis,
    cells: tuple[Cell, ...],
    mu: float,
    s: float,
    upper: float,
    lower: float,
    log_likelihood: float,
    link: str = "probit",
) -> Fit:
    params = {
        "mu": Parameter(value=mu, fixed=False),
        "s": Parameter(value=s, fixed=False),
        "upper": Parameter(value=upper, fixed=True),
        "lower": Parameter(value=lower, fixed=True),
    }
    covariance = Covariance(names=("mu", "s"), matrix=((0.01, 0.0), (0.0, 0.01)))
    return Fit(
        axis=axis,
        outcome="success",
        direction="decreasing",
        model="binomial",
        link=link,
        params=params,
        covariance=covariance,
        log_likelihood=log_likelihood,
        cells=cells,
        status=Status.OK,
    )


def _generic_fit(
    *,
    axis: Axis,
    cells: tuple[Cell, ...],
    mu: float,
    s: float,
    upper: float,
    upper_fixed: bool,
    lower: float,
    lower_fixed: bool,
    log_likelihood: float,
    link: str = "probit",
) -> Fit:
    """Like ``_fixed_fit``, but lets ``upper``/``lower`` each be estimated or fixed
    independently -- needed to test the "generic" (``GenericLikelihoodModel``) path's own
    exclusion behaviour directly, without depending on a real optimizer converging on a
    deliberately extreme, saturating dataset."""
    params = {
        "mu": Parameter(value=mu, fixed=False),
        "s": Parameter(value=s, fixed=False),
        "upper": Parameter(value=upper, fixed=upper_fixed),
        "lower": Parameter(value=lower, fixed=lower_fixed),
    }
    estimated = (
        ["mu", "s"]
        + (["upper"] if not upper_fixed else [])
        + (["lower"] if not lower_fixed else [])
    )
    n = len(estimated)
    matrix = tuple(tuple(0.01 if i == j else 0.0 for j in range(n)) for i in range(n))
    covariance = Covariance(names=tuple(estimated), matrix=matrix)
    return Fit(
        axis=axis,
        outcome="success",
        direction="decreasing",
        model="binomial",
        link=link,
        params=params,
        covariance=covariance,
        log_likelihood=log_likelihood,
        cells=cells,
        status=Status.OK,
    )


# ------------------------------------------------------------------------------------------
# Goodness of fit: a small case whose deviance/Pearson/df are computed directly, not read off
# diagnose()'s own internals (decisions/0023's "checked by a unit test that computes the
# formulas directly").
# ------------------------------------------------------------------------------------------


class TestGoodnessOfFitMatchesDirectFormula:
    def _build(self) -> tuple[Fit, list[float]]:
        mu, s, upper, lower = math.log(2.0), 0.5, 1.0, 0.0
        severities = [1.0, 2.0, 4.0]
        successes = [90, 55, 5]
        trials = [100, 100, 100]
        cells = tuple(
            Cell(severity=sv, successes=y, trials=n)
            for sv, y, n in zip(severities, successes, trials, strict=True)
        )
        p = [_probit_p(mu, s, upper, lower, sv) for sv in severities]
        ll = _loglik(cells, p)
        fit = _fixed_fit(
            axis=_LOG_AXIS, cells=cells, mu=mu, s=s, upper=upper, lower=lower, log_likelihood=ll
        )
        return fit, p

    def test_deviance_pearson_and_df_match_the_direct_formula(self) -> None:
        fit, p = self._build()
        expected_deviance, expected_pearson, expected_df = _direct_gof(fit.cells, p, k=2)

        diag = diagnose(fit)

        assert diag.df == expected_df
        assert diag.deviance == pytest.approx(expected_deviance, rel=1e-9)
        assert diag.pearson_chi2 == pytest.approx(expected_pearson, rel=1e-9)
        assert diag.dispersion == pytest.approx(expected_pearson / expected_df, rel=1e-9)


# ------------------------------------------------------------------------------------------
# C1 (decisions/0023 Amendment 2): the structural exclusion rule, pinned against the exact
# regression the stats review found -- df depending on the link because the pre-Amendment-2
# implementation excluded a cell by its *computed* fitted probability rather than by structure.
# ------------------------------------------------------------------------------------------


class TestStructuralDfExclusion:
    """Control + ``numpy.logspace(-3, 0, 10)``, n=400 per level, a success schedule chosen (by
    the stats review, not by this file) so probit and cloglog saturate a non-control cell to a
    computed fitted probability of exactly 0.0/1.0 in double precision while logit does not --
    the exact condition that made the pre-Amendment-2 implementation's df depend on the link (7
    for probit/cloglog, 8 for logit). The corrected, structural rule excludes only the control,
    giving df=8 for every link."""

    _SEVERITY = [0.0, *np.logspace(-3.0, 0.0, 10).tolist()]
    _SUCCESSES = [400, 400, 400, 400, 400, 399, 374, 179, 13, 1, 0]
    _TRIALS = [400] * 11

    def _obs(self) -> Observations:
        return Observations.from_counts(
            _LOG_AXIS,
            severity=self._SEVERITY,
            successes=self._SUCCESSES,
            trials=self._TRIALS,
            outcome="success",
            direction="decreasing",
        )

    @pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
    def test_df_is_8_for_every_link(self, link: str) -> None:
        fit = fit_dose_response(self._obs(), model="binomial", link=link, upper=1.0, lower=0.0)
        assert fit.status is Status.OK

        diag = diagnose(fit)

        assert diag.df == 8

    @pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
    def test_deviance_pearson_and_dispersion_are_all_finite(self, link: str) -> None:
        fit = fit_dose_response(self._obs(), model="binomial", link=link, upper=1.0, lower=0.0)
        assert fit.status is Status.OK

        diag = diagnose(fit)

        assert diag.deviance is not None and math.isfinite(diag.deviance)
        assert diag.pearson_chi2 is not None and math.isfinite(diag.pearson_chi2)
        assert diag.dispersion is not None and math.isfinite(diag.dispersion)


class TestControlCountedWhenUpperIsEstimated:
    """The structural exclusion requires ``upper`` to be *fixed* at exactly ``1.0`` -- not merely
    for the control's own fitted probability to equal ``1.0``. Built directly (bypassing
    ``fit_dose_response``, which does not converge with ``upper="estimate"`` on data this
    extreme): an *estimated* ``upper`` whose value happens to be exactly ``1.0`` still makes the
    control's fitted probability exactly ``1.0`` (R3's ``P(0) = upper`` rule applies regardless
    of ``fixed``), which is exactly the case the old, wrong "exclude by computed value" rule
    would still have dropped -- proving the fix checks ``fixed``, not the resulting number.
    """

    def test_control_cell_counts_towards_df_when_upper_is_estimated(self) -> None:
        mu, s, upper, lower = math.log(2.0), 0.5, 1.0, 0.0
        cells = (
            Cell(severity=0.0, successes=100, trials=100),  # p == upper == 1.0 exactly (R3)
            Cell(severity=1.0, successes=87, trials=100),
            Cell(severity=2.0, successes=48, trials=100),
            Cell(severity=4.0, successes=8, trials=100),
        )
        p = [upper] + [_probit_p(mu, s, upper, lower, c.severity) for c in cells[1:]]
        ll = _loglik(cells, p)
        fit = _generic_fit(
            axis=_LOG_AXIS,
            cells=cells,
            mu=mu,
            s=s,
            upper=upper,
            upper_fixed=False,  # <- estimated, even though its value is exactly 1.0
            lower=lower,
            lower_fixed=True,
            log_likelihood=ll,
        )

        diag = diagnose(fit)

        # 4 cells, all counted (control included) - 3 estimated params (mu, s, upper) = 1.
        # The old "exclude by computed value" rule would have dropped the control (p == 1.0),
        # giving 3 - 3 = 0 -> no GOF numbers at all.
        assert diag.df == 1
        assert diag.deviance is not None
        assert diag.pearson_chi2 is not None
        assert diag.dispersion is not None


class TestGenericPathSaturatedNonControlCellStillCounted:
    """ "with lower=0 fixed on the generic path a saturated cell is still counted" (decisions/0023
    Amendment 2). Built directly with an extreme severity (``z = 50``, where ``norm.cdf``
    saturates to exactly ``1.0`` in double precision) so the highest-severity cell's fitted
    probability is exactly ``lower = 0.0`` -- not the control, and not under a fixed ``upper`` --
    so the structural exclusion must not apply to it at all.
    """

    def test_saturated_high_severity_cell_still_counts_towards_df(self) -> None:
        mu, s, upper, lower = 0.0, 0.5, 0.9, 0.0
        # z = (log(severity) - mu) / s = 50 at this severity.
        saturated_severity = math.exp(25.0)
        severities = [0.5, 1.0, math.e, saturated_severity]
        p = [_probit_p(mu, s, upper, lower, sv) for sv in severities]
        # Confirms the fixture's own premise: exactly saturated, not merely small.
        assert p[-1] == 0.0
        cells = tuple(
            Cell(severity=sv, successes=round(100 * p_i), trials=100)
            for sv, p_i in zip(severities, p, strict=True)
        )
        ll = _loglik(cells, p)
        fit = _generic_fit(
            axis=_LOG_AXIS,
            cells=cells,
            mu=mu,
            s=s,
            upper=upper,
            upper_fixed=False,
            lower=lower,
            lower_fixed=True,
            log_likelihood=ll,
        )

        diag = diagnose(fit)

        # 4 cells, all counted - 3 estimated params (mu, s, upper) = 1. The old "exclude by
        # computed value" rule would have dropped the saturated cell, giving 3 - 3 = 0.
        assert diag.df == 1
        assert diag.deviance is not None
        assert diag.pearson_chi2 is not None
        assert diag.dispersion is not None


# ------------------------------------------------------------------------------------------
# Dispersion threshold: exactly 1.5 does not warn (plan section 5.7 / decisions/0023: "when phi
# > 1.5", strict), a hair above it does. Built with dyadic (exactly binary-representable)
# probabilities (0.75/0.5/0.25) so the boundary case lands on exactly 1.5 in floating point, not
# 1.4999999999999998 or 1.5000000000000002 -- this is a deliberate anti-flake measure, not an
# accident of the chosen numbers.
# ------------------------------------------------------------------------------------------


class TestDispersionThreshold:
    _MU = 1.0
    _S = 0.02
    _UPPER = 0.75
    _LOWER = 0.25
    _SEVERITIES = [0.0, 1.0, 2.0]  # z = -50, 0, +50 -> p = upper, midpoint, lower exactly

    def _cells(self, *, middle_successes: int) -> tuple[Cell, ...]:
        return (
            Cell(severity=0.0, successes=75, trials=100),
            Cell(severity=1.0, successes=middle_successes, trials=24),
            Cell(severity=2.0, successes=25, trials=100),
        )

    def _fit(self, *, middle_successes: int) -> Fit:
        cells = self._cells(middle_successes=middle_successes)
        p = [
            _probit_p_linear(self._MU, self._S, self._UPPER, self._LOWER, sv)
            for sv in self._SEVERITIES
        ]
        ll = _loglik(cells, p)
        return _fixed_fit(
            axis=_LINEAR_AXIS,
            cells=cells,
            mu=self._MU,
            s=self._S,
            upper=self._UPPER,
            lower=self._LOWER,
            log_likelihood=ll,
        )

    def test_dispersion_of_exactly_1_5_does_not_warn(self) -> None:
        fit = self._fit(middle_successes=15)  # pearson term = (15-12)^2/6 = 1.5 exactly

        diag = diagnose(fit)

        assert diag.pearson_chi2 == pytest.approx(1.5, abs=1e-9)
        assert diag.df == 1
        assert diag.dispersion == pytest.approx(1.5, abs=1e-9)
        assert not any(w.startswith("OVERDISPERSED: ") for w in diag.warnings)

    def test_dispersion_just_above_1_5_warns_with_the_overdispersed_token(self) -> None:
        fit = self._fit(middle_successes=16)  # pearson term = (16-12)^2/6 = 8/3

        diag = diagnose(fit)

        assert diag.dispersion is not None
        assert diag.dispersion > 1.5
        assert any(w.startswith("OVERDISPERSED: ") for w in diag.warnings)


# ------------------------------------------------------------------------------------------
# SPARSE_CELLS: (decisions/0023 Amendment 2 item 3, corrected by Amendment 3, cutoff tightened by
# Amendment 4) -- report-only, changes no number. A side (successes or failures) is flagged only
# when its expected count is below 0.1 AND its observed count is at least 1 -- a genuine event
# against a small expectation, not merely a small expectation on its own (which an all-success/
# all-failure plateau cell has on its "wrong" side too, harmlessly, per Amendment 3's own
# finding), and not merely *any* event against *any* sub-1 expectation (Amendment 3 alone still
# fired routinely on ordinary events at moderate expectations like ~0.43, per Amendment 4).
# ------------------------------------------------------------------------------------------


_SPARSE_FLOAT_RE = re.compile(r"-?\d+\.\d+(?:[eE][+-]?\d+)?")


def _named_severities(warning: str) -> list[str]:
    """The comma-separated severity list a ``SPARSE_CELLS: `` warning names, as literal
    substrings (not parsed to ``float``, so a formatting difference is still visible) -- pulled
    from between "severity/severities " and " have an expected count", the one piece of the
    message this file otherwise never depends on the exact wording of."""
    match = re.search(r"severity/severities (.+?) have an expected count", warning)
    assert match is not None, f"could not find a severity list in {warning!r}"
    return [token.strip() for token in match.group(1).split(",")]


class TestSparseCellsWarning:
    """Three cells, all fixed ``upper=1, lower=0``, chosen so the fitted probabilities are never
    near a saturating boundary (W1's "dominated by single events" finding is a different failure
    mode from C1's structural-exclusion bug above, and must not be confused with it -- none of
    these cells is ever a candidate for exclusion). decisions/0023 Amendment 4 tightened the
    expected-count cutoff from below 1 to below 0.1, so three roles are needed to pin the cutoff
    itself, not just "sparse or not":

    - ``0.09``: expected failures ``~= 0.096`` (**below** the 0.1 cutoff) and **one** observed
      failure (``99/100``) -- the genuine pathology: flagged.
    - ``0.05``: expected failures ``~= 0.011`` (also below the cutoff) but **zero** observed
      failures (all ``100/100`` success) -- Amendment 3's silencing case: a sparse side with
      nothing observed there is not flagged.
    - ``0.1522``: expected failures ``~= 0.50`` (**at or above** the 0.1 cutoff, though still
      below the old Amendment 3 cutoff of 1) with **one** observed failure (``99/100``) -- pins
      the cutoff's own boundary: this cell would have been flagged under Amendment 3 and must
      not be under Amendment 4.
    """

    _MU, _S, _UPPER, _LOWER = math.log(2.0), 1.0, 1.0, 0.0
    _FLAGGED_SEVERITY = 0.09  # E[fail] ~= 0.096 (< 0.1), 1 observed failure: flagged.
    _SILENT_ZERO_OBSERVED_SEVERITY = 0.05  # E[fail] ~= 0.011 (< 0.1), 0 observed: silent.
    _SILENT_ABOVE_CUTOFF_SEVERITY = 0.1522  # E[fail] ~= 0.50 (>= 0.1), 1 observed: silent.
    # Ascending in severity: ``Fit.cells`` must be strictly ascending (decisions/0029), so the
    # three roles are listed in that order and indexed below as 0 = zero-observed, 1 = flagged,
    # 2 = above-cutoff.
    _SEVERITIES = [
        _SILENT_ZERO_OBSERVED_SEVERITY,
        _FLAGGED_SEVERITY,
        _SILENT_ABOVE_CUTOFF_SEVERITY,
    ]
    _SUCCESSES = [100, 99, 99]
    _TRIALS = [100, 100, 100]

    def _cells(self) -> tuple[Cell, ...]:
        return tuple(
            Cell(severity=sv, successes=y, trials=n)
            for sv, y, n in zip(self._SEVERITIES, self._SUCCESSES, self._TRIALS, strict=True)
        )

    def _p(self) -> list[float]:
        return [
            _probit_p(self._MU, self._S, self._UPPER, self._LOWER, sv) for sv in self._SEVERITIES
        ]

    def _fit(self) -> Fit:
        cells = self._cells()
        p = self._p()
        ll = _loglik(cells, p)
        return _fixed_fit(
            axis=_LOG_AXIS,
            cells=cells,
            mu=self._MU,
            s=self._S,
            upper=self._UPPER,
            lower=self._LOWER,
            log_likelihood=ll,
        )

    def test_fixture_premise_pins_the_0_1_cutoff_and_the_observed_event_condition(self) -> None:
        p = self._p()
        n = 100
        # Flagged: expected failures below 0.1, with exactly 1 observed failure.
        assert n * (1.0 - p[1]) < 0.1
        assert n - self._SUCCESSES[1] == 1
        # Silent (zero observed): expected failures below 0.1, but 0 observed failures.
        assert n * (1.0 - p[0]) < 0.1
        assert n - self._SUCCESSES[0] == 0
        # Silent (at/above cutoff): expected failures in [0.1, 1), with 1 observed failure --
        # this is exactly the shape Amendment 3 alone would have flagged.
        assert 0.1 <= n * (1.0 - p[2]) < 1.0
        assert n - self._SUCCESSES[2] == 1

    def test_warning_fires_and_names_only_the_below_cutoff_severity_with_an_event(self) -> None:
        diag = diagnose(self._fit())

        sparse_warnings = [w for w in diag.warnings if w.startswith("SPARSE_CELLS: ")]
        assert len(sparse_warnings) == 1
        assert str(self._FLAGGED_SEVERITY) in sparse_warnings[0]

    def test_warning_does_not_name_the_severity_with_no_observed_event(self) -> None:
        """Amendment 3's own point, unchanged by Amendment 4: a small expected count on a side
        where *nothing was observed* is not a finding by itself."""
        diag = diagnose(self._fit())

        sparse_warning = next(w for w in diag.warnings if w.startswith("SPARSE_CELLS: "))
        assert str(self._SILENT_ZERO_OBSERVED_SEVERITY) not in sparse_warning

    def test_warning_does_not_name_the_severity_at_or_above_the_cutoff(self) -> None:
        """Amendment 4's own point: an observed event against an expectation of ~0.50 (below
        Amendment 3's old cutoff of 1, but at or above Amendment 4's 0.1) is no longer a
        finding."""
        diag = diagnose(self._fit())

        sparse_warning = next(w for w in diag.warnings if w.startswith("SPARSE_CELLS: "))
        assert str(self._SILENT_ABOVE_CUTOFF_SEVERITY) not in sparse_warning

    def test_minimum_expected_count_in_the_message_is_over_flagged_sides_only(self) -> None:
        """Only one side is flagged here, so the minimum in the message must be that side's own
        expected count (~=0.096) -- not the smaller, unflagged ~=0.011 silent side, and not the
        larger, unflagged ~=0.50 above-cutoff side."""
        diag = diagnose(self._fit())

        sparse_warning = next(w for w in diag.warnings if w.startswith("SPARSE_CELLS: "))
        p = self._p()
        flagged_expected = 100 * (1.0 - p[1])
        other_expecteds = [100 * (1.0 - p[0]), 100 * (1.0 - p[2])]

        numbers = [float(m) for m in _SPARSE_FLOAT_RE.findall(sparse_warning)]
        assert any(abs(n - flagged_expected) < 0.01 for n in numbers), (
            f"expected a number near {flagged_expected!r} (the only flagged side's own expected "
            f"count) in {sparse_warning!r}"
        )
        for other in other_expecteds:
            assert not any(abs(n - other) < 0.01 for n in numbers), (
                f"an unflagged side's expected count {other!r} must not appear as the reported "
                f"minimum in {sparse_warning!r}"
            )

    def test_warning_changes_no_number(self) -> None:
        """The warning is report-only (Amendment 2, item 3): deviance/pearson_chi2/df/dispersion
        equal the plain formula, computed independently, with no special-casing for sparseness."""
        cells = self._cells()
        p = self._p()
        expected_deviance, expected_pearson, expected_df = _direct_gof(cells, p, k=2)

        diag = diagnose(self._fit())

        assert diag.df == expected_df
        assert diag.deviance == pytest.approx(expected_deviance, rel=1e-9)
        assert diag.pearson_chi2 == pytest.approx(expected_pearson, rel=1e-9)
        assert diag.dispersion == pytest.approx(expected_pearson / expected_df, rel=1e-9)

    def test_does_not_fire_on_a_well_behaved_dataset(self) -> None:
        """Every expected count is well above 1 here (the same fixture
        ``TestGoodnessOfFitMatchesDirectFormula`` uses) -- no cell is sparse."""
        mu, s, upper, lower = math.log(2.0), 0.5, 1.0, 0.0
        severities = [1.0, 2.0, 4.0]
        successes = [90, 55, 5]
        trials = [100, 100, 100]
        cells = tuple(
            Cell(severity=sv, successes=y, trials=n)
            for sv, y, n in zip(severities, successes, trials, strict=True)
        )
        p = [_probit_p(mu, s, upper, lower, sv) for sv in severities]
        assert all(n * p_i >= 1.0 and n * (1.0 - p_i) >= 1.0 for p_i in p for n in [100])
        ll = _loglik(cells, p)
        fit = _fixed_fit(
            axis=_LOG_AXIS, cells=cells, mu=mu, s=s, upper=upper, lower=lower, log_likelihood=ll
        )

        diag = diagnose(fit)

        assert not any(w.startswith("SPARSE_CELLS: ") for w in diag.warnings)


class TestSparseCellsSilentOnCleanPlateaus:
    """A correctly specified fit on the rich design (control + ``logspace(-2, 2, 10)``, n=400,
    probit, ``mu=0``, ``s=0.5``) must never warn -- the blanket claim decisions/0023 Amendment 4
    restores. Amendment 3 alone (cutoff below 1) still fired on about 57.5% of such fits, because
    two grid points near the transition (about 0.215 and 4.64) expect about 0.43 events on their
    sparse side, and observing one there is ordinary, not a finding -- this file's own
    ``test-author`` re-verification found 11 of 20 seeded replicates fired under Amendment 3
    alone, which is what led to Amendment 4's tightened 0.1 cutoff: at 0.1, neither near-transition
    cell qualifies at all (0.43 is not below 0.1), so nothing here can legitimately fire any more.
    """

    _MU, _S, _UPPER, _LOWER = 0.0, 0.5, 1.0, 0.0
    _SEVERITY = [0.0, *np.logspace(-2.0, 2.0, 10).tolist()]
    _TRIALS = [400] * 11

    def _expected_p(self) -> list[float]:
        p = [self._UPPER]  # control, R3
        for sv in self._SEVERITY[1:]:
            p.append(_probit_p(self._MU, self._S, self._UPPER, self._LOWER, sv))
        return p

    def test_deterministic_rounded_plateau_fit_never_warns(self) -> None:
        """Successes are ``round(n * p)`` at the *true* curve -- not sampled -- so every cell
        whose true expected count on a side is below 1 rounds to exactly 0 observed there,
        deterministically and regardless of any seed. This is the strongest possible version of
        "silent on clean plateaus": it holds with certainty, not merely with high probability."""
        p = self._expected_p()
        successes = [round(n * p_i) for n, p_i in zip(self._TRIALS, p, strict=True)]

        obs = Observations.from_counts(
            _LOG_AXIS,
            severity=self._SEVERITY,
            successes=successes,
            trials=self._TRIALS,
            outcome="success",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
        assert fit.status is Status.OK

        diag = diagnose(fit)

        assert not any(w.startswith("SPARSE_CELLS: ") for w in diag.warnings)

    def test_seeded_replicates_never_warn_at_all(self) -> None:
        """20 real (sampled, not rounded) seeded replicates on the same design: at the 0.1
        cutoff, no cell -- including the two near-transition ones -- can legitimately fire any
        more (``test_fixture_premise_the_two_near_transition_cells_no_longer_qualify`` below
        confirms their expected counts sit at or above 0.1, deterministically, regardless of
        sampling), so the blanket claim is restored: no ``SPARSE_CELLS:`` warning at all, on any
        of the 20 replicates.
        """
        rng = np.random.default_rng(20260928)
        p = self._expected_p()

        for i in range(20):
            successes = rng.binomial(self._TRIALS, p)
            obs = Observations.from_counts(
                _LOG_AXIS,
                severity=self._SEVERITY,
                successes=successes.tolist(),
                trials=self._TRIALS,
                outcome="success",
                direction="decreasing",
            )
            fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
            if fit.status is not Status.OK:
                continue
            diag = diagnose(fit)
            sparse = [w for w in diag.warnings if w.startswith("SPARSE_CELLS: ")]
            assert not sparse, f"replicate {i}: unexpected SPARSE_CELLS warning: {sparse!r}"

    def test_fixture_premise_the_two_near_transition_cells_no_longer_qualify(self) -> None:
        """decisions/0023 Amendment 4's own rationale: the two near-transition points (severities
        ~0.2154 and ~4.6416) have an expected count of about 0.43 on their sparse side --
        comfortably at or above the 0.1 cutoff, so no observed count there can ever flag them,
        regardless of sampling. (Under the superseded Amendment-3-only cutoff of 1, these same two
        cells *did* legitimately qualify -- confirmed empirically during that amendment's own
        development, seed 20260928, 11 of 20 replicates.)"""
        p = self._expected_p()
        near_transition_indices = (4, 7)  # severities ~0.2154 and ~4.6416
        for idx in near_transition_indices:
            p_i = p[idx]
            n = self._TRIALS[idx]
            sparse_side_expected = n * min(p_i, 1.0 - p_i)
            assert sparse_side_expected >= 0.1, (
                f"severity {self._SEVERITY[idx]}: expected this cell's sparse side to sit at or "
                f"above the 0.1 cutoff, got {sparse_side_expected!r}"
            )


class TestSparseCellsSilentOnC1GridPlateauCells:
    """The C1 saturating grid (``TestStructuralDfExclusion``'s own fixture), re-checked under
    Amendment 4's 0.1 cutoff, link by link (each link fits different ``mu``/``s``, so which cells
    end up below the cutoff differs per link -- this was re-verified directly, not assumed):

    - **probit, logit:** the two cells that had a genuine single event against a ~0.39-0.40 or
      ~0.15 expectation under the old (Amendment-3-only) cutoff of 1 no longer qualify at 0.1 --
      neither link flags *any* cell on this grid.
    - **cloglog:** severity ``~0.4642`` (1 observed success where cloglog's own fit puts the
      expectation at about 2e-6, far below 0.1) still flags -- cloglog's asymmetry fits this
      saturating grid differently enough that this single success is a genuine, extreme outlier
      under its own curve, not merely an artifact of the cutoff.
    """

    _SEVERITY = [0.0, *np.logspace(-3.0, 0.0, 10).tolist()]
    _SUCCESSES = [400, 400, 400, 400, 400, 399, 374, 179, 13, 1, 0]
    _TRIALS = [400] * 11

    def _obs(self) -> Observations:
        return Observations.from_counts(
            _LOG_AXIS,
            severity=self._SEVERITY,
            successes=self._SUCCESSES,
            trials=self._TRIALS,
            outcome="success",
            direction="decreasing",
        )

    @pytest.mark.parametrize("link", ["probit", "logit"])
    def test_no_cell_is_flagged(self, link: str) -> None:
        fit = fit_dose_response(self._obs(), model="binomial", link=link, upper=1.0, lower=0.0)
        assert fit.status is Status.OK

        diag = diagnose(fit)

        assert not any(w.startswith("SPARSE_CELLS: ") for w in diag.warnings)

    def test_cloglog_flags_exactly_its_own_extreme_single_success(self) -> None:
        """Asserts the *exact* named severity, not merely that it is present: under the
        superseded Amendment-3-only cutoff of 1, cloglog's own warning here also (wrongly,
        post-Amendment-4) names four plateau severities (the four all-success cells and the
        all-failure one) that must be silent at the 0.1 cutoff -- a looser "is present" check
        would not have caught that regression.

        The expected severity is derived from the fixture's own arrays (the cell whose observed
        successes is exactly 1), not hard-coded as a computed ``np.logspace`` literal: a
        platform's own libm can differ from this machine's in the last bit of such a value (CI
        found exactly that -- ``0.46415888336127775`` here, ``0.4641588833612777`` on another
        runner, both correct), so the expected string must be ``repr()`` of the *same* float
        object the fixture already built, formatted exactly the way ``validation.py`` formats a
        severity in the message (plain ``repr``, checked directly against its source)."""
        flagged_index = self._SUCCESSES.index(1)  # the one cell with exactly 1 observed success
        expected_severity_repr = repr(self._SEVERITY[flagged_index])

        fit = fit_dose_response(self._obs(), model="binomial", link="cloglog", upper=1.0, lower=0.0)
        assert fit.status is Status.OK

        diag = diagnose(fit)
        sparse_warnings = [w for w in diag.warnings if w.startswith("SPARSE_CELLS: ")]

        assert len(sparse_warnings) == 1
        assert _named_severities(sparse_warnings[0]) == [expected_severity_repr], sparse_warnings[0]


class TestSparseCellsFiresOnZetaPidSensorNoise:
    """decisions/0023 Amendment 2's own motivating real-data case, re-confirmed under Amendment
    4's 0.1 cutoff: 1 observed failure at severity 0.01 where roughly 0.005 (probit) / 0.015
    (logit) were expected -- both comfortably below 0.1, a genuine event against a small
    expectation, so it must still fire. Under cloglog the same cell's own expectation is about
    0.95 (well above the cutoff), so it must not fire there -- the cutoff, not the token, is what
    makes the difference (decisions/0023 Amendment 4's own recorded consequence)."""

    _SEVERITY = [0.0, 0.01, 0.05, 0.1]
    _SUCCESSES = [200, 299, 122, 0]
    _TRIALS = [200, 300, 300, 300]

    def _obs(self) -> Observations:
        axis = Axis(name="sensor_noise", unit="m", scale="log")
        return Observations.from_counts(
            axis,
            severity=self._SEVERITY,
            successes=self._SUCCESSES,
            trials=self._TRIALS,
            outcome="success",
            direction="decreasing",
        )

    @pytest.mark.parametrize("link", ["probit", "logit"])
    def test_warning_fires_and_names_0_01(self, link: str) -> None:
        fit = fit_dose_response(self._obs(), model="binomial", link=link, upper=1.0, lower=0.0)
        assert fit.status is Status.OK

        diag = diagnose(fit)
        sparse_warnings = [w for w in diag.warnings if w.startswith("SPARSE_CELLS: ")]

        assert sparse_warnings, f"{link}: expected a SPARSE_CELLS warning"
        assert "0.01" in sparse_warnings[0]

    def test_cloglog_does_not_fire(self) -> None:
        fit = fit_dose_response(self._obs(), model="binomial", link="cloglog", upper=1.0, lower=0.0)
        assert fit.status is Status.OK

        diag = diagnose(fit)

        assert not any(w.startswith("SPARSE_CELLS: ") for w in diag.warnings)


# ------------------------------------------------------------------------------------------
# df <= 0: not a failed fit, but no GOF numbers either (decisions/0023).
# ------------------------------------------------------------------------------------------


class TestDegreesOfFreedomAtOrBelowZero:
    def _fit(self, cells: tuple[Cell, ...]) -> Fit:
        p = [_probit_p(0.0, 1.0, 1.0, 0.0, c.severity) for c in cells]
        ll = _loglik(cells, p)
        return _fixed_fit(
            axis=_LOG_AXIS, cells=cells, mu=0.0, s=1.0, upper=1.0, lower=0.0, log_likelihood=ll
        )

    def test_exactly_saturated_df_zero_gives_none_stats_and_a_warning(self) -> None:
        # 2 cells, 2 estimated params (mu, s; upper/lower fixed) -> df = 0.
        cells = (
            Cell(severity=1.0, successes=6, trials=10),
            Cell(severity=2.0, successes=3, trials=10),
        )
        fit = self._fit(cells)

        diag = diagnose(fit)

        assert diag.df == 0
        assert diag.deviance is None
        assert diag.pearson_chi2 is None
        assert diag.dispersion is None
        assert diag.warnings
        assert any("df" in w.lower() or "degrees of freedom" in w.lower() for w in diag.warnings)

    def test_negative_df_also_gives_none_stats_not_a_crash(self) -> None:
        # 1 cell, 2 estimated params -> df = -1.
        cells = (Cell(severity=1.0, successes=6, trials=10),)
        fit = self._fit(cells)

        diag = diagnose(fit)

        assert diag.df == -1
        assert diag.deviance is None
        assert diag.pearson_chi2 is None
        assert diag.dispersion is None

    def test_a_non_ok_fit_status_is_not_produced_by_a_low_df_warning(self) -> None:
        """decisions/0023: df <= 0 is *not* a failed fit -- the ``Fit`` itself stays whatever it
        already was (``OK`` here); only ``Diagnostics`` carries the warning."""
        cells = (
            Cell(severity=1.0, successes=6, trials=10),
            Cell(severity=2.0, successes=3, trials=10),
        )
        fit = self._fit(cells)

        diagnose(fit)

        assert fit.status is Status.OK


# ------------------------------------------------------------------------------------------
# A failed fit carries no diagnostics: fit_dose_response only fills Fit.diagnostics for
# Status.OK (decisions/0023, hard constraint 3). Monotonicity is the mirror-image case --
# populated on every status -- and is covered in tests/unit/test_monotonicity.py, not here.
# ------------------------------------------------------------------------------------------


class TestFailedFitHasNoDiagnostics:
    def test_control_incompatible_fit_has_diagnostics_none(self) -> None:
        axis = Axis(name="sensor_noise", unit="m", scale="log")
        obs = Observations.from_counts(
            axis,
            severity=[0.0, 0.01, 0.05, 0.1],
            successes=[57, 143, 0, 0],
            trials=[200, 300, 300, 300],
            outcome="success",
            direction="decreasing",
        )

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        assert fit.status is Status.CONTROL_INCOMPATIBLE
        assert fit.diagnostics is None

    def test_separation_fit_has_diagnostics_none(self) -> None:
        axis = Axis(name="wind", unit="m/s", scale="linear")
        obs = Observations.from_counts(
            axis,
            severity=[0.0, 2.0, 5.0, 10.0],
            successes=[100, 500, 500, 500],
            trials=[100, 500, 500, 500],
            outcome="success",
            direction="decreasing",
        )

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        assert fit.status is Status.SEPARATION
        assert fit.diagnostics is None


# ------------------------------------------------------------------------------------------
# Link comparison: each link's AIC equals -2*loglik + 2*k of that link's *own* fit under the
# same upper/lower spec; a link whose own refit does not converge reports AIC=None and its
# Status.
#
# The dataset below replaces an earlier one "found by search" (severity=[2,3,10],
# successes=[10,4,3], trials=[15,5,5]) that pinned cloglog's own NOT_CONVERGED as a fixed
# literal: 3 cells against 3 estimated parameters (mu, s, upper) put convergence on a knife
# edge that flipped with platform and library versions (CI found py3.11/Linux converging
# probit and logit too at latest dependencies, and cloglog converging instead at the
# minimum-versions floor) -- a test-design flaw, not a statistical finding. The dataset here is
# deliberately over-determined instead: 6 cells (including a zero-severity control anchoring the
# estimated ``upper``), n=200 per cell, a smooth monotone success schedule spanning the full
# dynamic range with no cell near a separation or sparse-count boundary -- verified locally to
# converge for all three links with comfortable margin, not merely found to converge once.
# ------------------------------------------------------------------------------------------


class TestLinkComparison:
    _SEVERITY = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
    _SUCCESSES = [190, 166, 136, 95, 54, 24]
    _TRIALS = [200] * 6

    def _obs(self) -> Observations:
        axis = Axis(name="severity", unit="unit", scale="log")
        return Observations.from_counts(
            axis,
            severity=self._SEVERITY,
            successes=self._SUCCESSES,
            trials=self._TRIALS,
            outcome="success",
            direction="decreasing",
        )

    def test_setup_sanity_all_three_links_converge(self) -> None:
        """Confirms the fixture's own premise before trusting the assertions built on it."""
        obs = self._obs()
        statuses = {
            link: fit_dose_response(
                obs, model="binomial", link=link, upper="estimate", lower=0.0
            ).status
            for link in ("probit", "logit", "cloglog")
        }

        assert statuses == {
            "probit": Status.OK,
            "logit": Status.OK,
            "cloglog": Status.OK,
        }

    def _fit(self, link: str) -> Fit:
        return fit_dose_response(
            self._obs(), model="binomial", link=link, upper="estimate", lower=0.0
        )

    def test_each_converged_links_aic_matches_its_own_fit(self) -> None:
        # All three links converge on this six-cell design (upper estimated at ~0.95), so the
        # refit-parity check covers all three, not just logit.
        fit_probit = self._fit("probit")
        fit_logit = self._fit("logit")
        fit_cloglog = self._fit("cloglog")
        assert fit_probit.status is Status.OK
        assert fit_logit.status is Status.OK
        assert fit_cloglog.status is Status.OK
        k = 3  # mu, s, upper estimated; lower fixed at 0.0

        diag = diagnose(fit_probit)

        assert fit_probit.log_likelihood is not None
        assert fit_logit.log_likelihood is not None
        assert fit_cloglog.log_likelihood is not None
        assert diag.link_aic["probit"] == pytest.approx(
            -2.0 * fit_probit.log_likelihood + 2.0 * k, rel=1e-9
        )
        assert diag.link_aic["logit"] == pytest.approx(
            -2.0 * fit_logit.log_likelihood + 2.0 * k, rel=1e-9
        )
        assert diag.link_aic["cloglog"] == pytest.approx(
            -2.0 * fit_cloglog.log_likelihood + 2.0 * k, rel=1e-9
        )
        assert diag.link_status["probit"] is Status.OK
        assert diag.link_status["logit"] is Status.OK
        assert diag.link_status["cloglog"] is Status.OK

    def test_a_non_converged_alternative_link_reports_aic_none_and_its_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The alternative-link refit's own non-convergence is forced deterministically here
        (rather than found by search) by monkeypatching the fitting call
        ``validation.py``'s ``_refit_link`` uses for any non-fixed-asymptote spec --
        ``robustmargin.models._fit_generic`` -- so only the ``cloglog`` refit is intercepted and
        every other call (including the ``fit_probit`` fit this test itself builds, and the real
        ``logit`` refit ``diagnose`` also performs) goes through unchanged. This tests the same
        ``_link_comparison`` code path as before, just without depending on an optimizer landing
        on the correct side of a tolerance."""
        import robustmargin.models as models_module

        real_fit_generic = models_module._fit_generic

        def _fake_fit_generic(obs: Observations, **kwargs: object) -> Fit:
            if kwargs.get("link") == "cloglog":
                return models_module._status_only_fit(
                    obs,
                    cells=kwargs["cells"],  # type: ignore[arg-type]
                    cluster_ids=kwargs["cluster_ids"],  # type: ignore[arg-type]
                    link="cloglog",
                    status=Status.NOT_CONVERGED,
                    fit_warnings=("test: forced NOT_CONVERGED for deterministic CI coverage",),
                )
            return real_fit_generic(obs, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(models_module, "_fit_generic", _fake_fit_generic)

        fit_probit = self._fit("probit")
        assert fit_probit.status is Status.OK

        diag = diagnose(fit_probit)

        assert diag.link_aic["cloglog"] is None
        assert diag.link_status["cloglog"] is Status.NOT_CONVERGED
        # The patch did not disturb the real, converging logit refit.
        assert diag.link_status["logit"] is Status.OK

    def test_link_aic_and_link_status_cover_exactly_the_three_links(self) -> None:
        fit_probit = self._fit("probit")

        diag = diagnose(fit_probit)

        assert set(diag.link_aic) == {"probit", "logit", "cloglog"}
        assert set(diag.link_status) == {"probit", "logit", "cloglog"}


# ------------------------------------------------------------------------------------------
# diagnose() never changes the fit it was given (decisions/0023: "Diagnostics never change a
# result").
# ------------------------------------------------------------------------------------------


class TestDiagnoseDoesNotMutateTheFit:
    def test_fit_link_is_unchanged_after_diagnose(self) -> None:
        mu, s, upper, lower = math.log(2.0), 0.5, 1.0, 0.0
        cells = (
            Cell(severity=1.0, successes=90, trials=100),
            Cell(severity=2.0, successes=55, trials=100),
            Cell(severity=4.0, successes=5, trials=100),
        )
        p = [_probit_p(mu, s, upper, lower, c.severity) for c in cells]
        ll = _loglik(cells, p)
        fit = _fixed_fit(
            axis=_LOG_AXIS, cells=cells, mu=mu, s=s, upper=upper, lower=lower, log_likelihood=ll
        )
        original_link = fit.link

        diagnose(fit)

        assert fit.link == original_link == "probit"

    def test_diagnostics_is_a_frozen_dataclass_instance(self) -> None:
        mu, s, upper, lower = math.log(2.0), 0.5, 1.0, 0.0
        cells = (
            Cell(severity=1.0, successes=90, trials=100),
            Cell(severity=2.0, successes=55, trials=100),
            Cell(severity=4.0, successes=5, trials=100),
        )
        p = [_probit_p(mu, s, upper, lower, c.severity) for c in cells]
        ll = _loglik(cells, p)
        fit = _fixed_fit(
            axis=_LOG_AXIS, cells=cells, mu=mu, s=s, upper=upper, lower=lower, log_likelihood=ll
        )

        diag = diagnose(fit)

        assert isinstance(diag, Diagnostics)
        with pytest.raises(dataclasses.FrozenInstanceError):
            diag.df = 99  # type: ignore[misc]
