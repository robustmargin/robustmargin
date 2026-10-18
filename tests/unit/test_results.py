"""Unit tests for ``Fit``, ``Threshold`` and ``Ratio``: the invariants each dataclass enforces in
its ``__post_init__``.

One rejection per rule ("one behaviour per test"), so a loosened validator shows up as one
specific failing test rather than a generic one. Also covers ``Parameter`` and ``Covariance``
(``models.py``), which have no rules of their own beyond ``Covariance``'s shape checks.

``Fit`` carries ``outcome`` and ``direction`` (``axis, outcome, direction, model, link, params,
covariance, log_likelihood, cells, status, warnings, schema_version, provenance``).

Threshold and Ratio's censoring/status combinations are deliberately only tested against the
rules the dataclasses actually state; ratio censoring propagation is ``ratio_interval``'s
concern, and this file does not invent rules for it.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from robustmargin import (
    Axis,
    Cell,
    Censoring,
    Covariance,
    Definition,
    Fit,
    IntervalShape,
    Parameter,
    Ratio,
    Status,
    Threshold,
)

_AXIS = Axis(name="noise", unit="m", scale="log")
_OTHER_NAME_AXIS = Axis(name="axis_2", unit="m", scale="linear")
_OTHER_UNIT_AXIS = Axis(name="noise", unit="dB", scale="log")
_CELLS = (
    Cell(severity=0.0, successes=100, trials=100),
    Cell(severity=1.0, successes=40, trials=100),
)
# A second set of cells. Two disjoint samples can produce identical counts and so identical fits
# (a ratio between them is valid, and is not rejected), but distinct default fits are the clearer
# fixture. This set lets _ratio()'s own default threshold_a/threshold_b -- and every helper below
# that builds a t_a/t_b pair from otherwise-identical arguments -- produce two genuinely distinct
# Fit objects instead of two that merely happen to compare equal. The zero-severity control stays
# an all-success 100/100, matching _CELLS's: _params()'s default fixes upper=1.0, and a *failing*
# control combined with a fixed upper=1.0 on a log axis is CONTROL_INCOMPATIBLE territory, which
# would raise here for an unrelated reason.
_CELLS_B = (
    Cell(severity=0.0, successes=100, trials=100),
    Cell(severity=1.0, successes=35, trials=100),
)


def _params(
    *, mu: float = 0.0, s: float = 1.0, upper: float = 1.0, lower: float = 0.0
) -> dict[str, Parameter]:
    return {
        "mu": Parameter(value=mu, fixed=False),
        "s": Parameter(value=s, fixed=False),
        "upper": Parameter(value=upper, fixed=True),
        "lower": Parameter(value=lower, fixed=True),
    }


def _covariance(names: tuple[str, ...] = ("mu", "s")) -> Covariance:
    n = len(names)
    matrix = tuple(tuple(1.0 if i == j else 0.0 for j in range(n)) for i in range(n))
    return Covariance(names=names, matrix=matrix)


def _fit_with_axis(
    axis: Axis,
    *,
    status: Status = Status.OK,
    direction: str = "decreasing",
    cells: tuple[Cell, ...] = _CELLS,
) -> Fit:
    if status is Status.OK:
        return Fit(
            axis=axis,
            outcome="success",
            direction=direction,
            model="binomial",
            link="probit",
            params=_params(),
            covariance=_covariance(),
            log_likelihood=-42.0,
            cells=cells,
            status=Status.OK,
        )
    return Fit(
        axis=axis,
        outcome="success",
        direction=direction,
        model="binomial",
        link="probit",
        params=None,
        covariance=None,
        log_likelihood=None,
        cells=cells,
        status=status,
    )


def _ok_fit() -> Fit:
    return _fit_with_axis(_AXIS)


# Which Fit.status a Threshold's own default fit should carry, mirroring
# robustmargin.testing._FIT_STATUS_FOR_THRESHOLD_STATUS. UNREACHABLE and FAILS_AT_BASELINE are
# properties of a perfectly good fit, so they map to OK; SEPARATION, NOT_CONVERGED
# and CONTROL_INCOMPATIBLE are properties of the fit itself, so a threshold with one of those
# statuses is built by default on a fit carrying the same status. A test that wants to
# exercise a *mismatched* fit status passes ``fit=`` explicitly, overriding this default.
_FIT_STATUS_FOR_THRESHOLD_STATUS: dict[Status, Status] = {
    Status.OK: Status.OK,
    Status.UNREACHABLE: Status.OK,
    Status.FAILS_AT_BASELINE: Status.OK,
    Status.SEPARATION: Status.SEPARATION,
    Status.NOT_CONVERGED: Status.NOT_CONVERGED,
    Status.CONTROL_INCOMPATIBLE: Status.CONTROL_INCOMPATIBLE,
}


def _threshold(
    *,
    value: float | None,
    lo: float | None,
    hi: float | None,
    level: float = 0.95,
    interval_method: str = "profile",
    censoring: Censoring = Censoring.NONE,
    status: Status = Status.OK,
    axis: Axis = _AXIS,
    direction: str = "decreasing",
    definition: Definition | None = None,
    dependence: str = "independent",
    fit: Fit | None = None,
) -> Threshold:
    # Threshold has no n_per_level -- per-level n is read from threshold.fit.cells instead, so
    # nothing is passed for it here.
    default_fit_status = _FIT_STATUS_FOR_THRESHOLD_STATUS[status]
    return Threshold(
        axis=axis,
        definition=definition if definition is not None else Definition.absolute(0.95),
        direction=direction,
        value=value,
        lo=lo,
        hi=hi,
        level=level,
        interval_method=interval_method,
        censoring=censoring,
        status=status,
        dependence=dependence,
        fit=fit
        if fit is not None
        else _fit_with_axis(axis, status=default_fit_status, direction=direction),
    )


def _ratio(
    *,
    estimate: float | None,
    lo: float | None,
    hi: float | None,
    shape: IntervalShape | None,
    status: Status = Status.OK,
    censoring: Censoring = Censoring.NONE,
    method: str = "fieller",
    dependence: str = "independent",
    level: float = 0.95,
    threshold_a: Threshold | None = None,
    threshold_b: Threshold | None = None,
) -> Ratio:
    # Both default thresholds carry the same value (0.05), so the default ratio is exactly 1.0 --
    # satisfying the rule that the estimate must equal threshold_a.value / threshold_b.value for
    # every call site that passes estimate=1.0 without overriding threshold_a/threshold_b. Disjoint
    # data between a ratio's two thresholds is robustmargin.testing.fake_ratio's concern, not this
    # dataclass-level helper's; nothing here tests independence.
    #
    # The two default fits are built from different cells (_CELLS vs _CELLS_B), modelling
    # disjoint data, even though the *threshold* values (0.05 / 0.05) match.
    return Ratio(
        threshold_a=threshold_a
        if threshold_a is not None
        else _threshold(
            value=0.05,
            lo=0.03,
            hi=0.08,
            interval_method="profile",
            fit=_fit_with_axis(_AXIS, cells=_CELLS),
        ),
        threshold_b=threshold_b
        if threshold_b is not None
        else _threshold(
            value=0.05,
            lo=0.03,
            hi=0.08,
            interval_method="profile",
            fit=_fit_with_axis(_AXIS, cells=_CELLS_B),
        ),
        estimate=estimate,
        lo=lo,
        hi=hi,
        shape=shape,
        method=method,
        dependence=dependence,
        level=level,
        censoring=censoring,
        status=status,
    )


class TestParameter:
    def test_valid_construction(self) -> None:
        parameter = Parameter(value=0.5, fixed=True)

        assert parameter.value == 0.5
        assert parameter.fixed is True

    def test_is_frozen(self) -> None:
        parameter = Parameter(value=0.5, fixed=False)

        with pytest.raises(dataclasses.FrozenInstanceError):
            cast(Any, parameter).value = 1.0

    def test_bool_value_raises_value_error(self) -> None:
        """A bool is not a float, even though ``isinstance(True, float)`` is False but
        ``math.isfinite(True)`` is True -- the existing finiteness check alone does not catch
        this."""
        with pytest.raises(ValueError):
            Parameter(value=True, fixed=False)

    def test_non_bool_fixed_raises_value_error(self) -> None:
        """``fixed`` must be a real ``bool``, not merely truthy."""
        with pytest.raises(ValueError):
            Parameter(value=0.5, fixed=1)


class TestCovariance:
    def test_valid_construction(self) -> None:
        covariance = Covariance(names=("mu", "s"), matrix=((1.0, 0.0), (0.0, 1.0)))

        assert covariance.names == ("mu", "s")
        assert covariance.matrix == ((1.0, 0.0), (0.0, 1.0))

    def test_non_square_matrix_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Covariance(names=("mu", "s"), matrix=((1.0, 0.0),))

    def test_matrix_side_must_match_the_number_of_names(self) -> None:
        with pytest.raises(ValueError):
            Covariance(names=("mu",), matrix=((1.0, 0.0), (0.0, 1.0)))

    def test_non_finite_entry_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Covariance(names=("mu", "s"), matrix=((1.0, 0.0), (0.0, float("inf"))))

    def test_non_symmetric_matrix_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Covariance(names=("mu", "s"), matrix=((1.0, 0.5), (0.2, 1.0)))

    def test_non_positive_semi_definite_matrix_raises_value_error(self) -> None:
        """Every diagonal entry must be positive, and the correlation matrix
        ``D^-1/2 . matrix . D^-1/2`` must have minimum eigenvalue ``> 1e-12`` (see the
        units-independence section below, in particular ``TestCovarianceStillRejects``, for the
        rule's own dedicated coverage). This matrix's first diagonal entry is ``-1.0``, which
        alone fails the rule (its eigenvalues, exactly -1 and 1, would fail a raw-eigenvalue
        rule too, but the diagonal check rejects it first)."""
        with pytest.raises(ValueError):
            Covariance(names=("mu", "s"), matrix=((-1.0, 0.0), (0.0, 1.0)))

    def test_is_frozen(self) -> None:
        covariance = Covariance(names=("mu",), matrix=((1.0,),))

        with pytest.raises(dataclasses.FrozenInstanceError):
            cast(Any, covariance).names = ("s",)


# ------------------------------------------------------------------------------------------
# Covariance's positive-definiteness rule is units-independent. A raw-eigenvalue rule (min
# eig(matrix) > 1e-12 * max(1, max|entry|)) would make an estimated-asymptote fit's status depend on
# the caller's severity units, because it mixes severity-scale parameters (mu, s) with probabilities
# (upper, lower) on the *raw* eigenvalue scale. The rule (`models._is_positive_definite`) tests the
# correlation matrix ``D^-1/2 . matrix . D^-1/2`` instead: every diagonal entry positive, and the
# correlation matrix's minimum eigenvalue > 1e-12. This section is that rule's own dedicated
# coverage, independent of any particular Fit or fitting path.
# ------------------------------------------------------------------------------------------


def _raw_eigenvalue_rule_accepts(matrix: np.ndarray) -> bool:
    """The scale-sensitive raw-eigenvalue positive-definiteness rule, implemented here
    independently (not imported from ``robustmargin.models``) as the oracle for
    ``TestCovarianceAcceptsEveryMatrixTheRawEigenvalueRuleAccepts`` below: ``min eig(matrix) >
    1e-12 * max(1, max|entry|)``, on the raw (non-correlation) matrix.
    """
    max_abs = float(np.max(np.abs(matrix)))
    tolerance = 1e-12 * max(1.0, max_abs)
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(matrix)))
    return min_eigenvalue > tolerance


def _covariance_accepts(matrix: np.ndarray) -> bool:
    """Whether the public rule accepts ``matrix``, decided the only way a test is
    allowed to decide it: by actually constructing a :class:`~robustmargin.Covariance` and seeing
    whether it raises. This never reimplements or imports ``models._is_positive_definite``
    itself, so the property test below exercises the real public behaviour, not a second copy
    of it.
    """
    n = matrix.shape[0]
    names = tuple(f"p{i}" for i in range(n))
    try:
        Covariance(names=names, matrix=tuple(tuple(row) for row in matrix))
    except ValueError:
        return False
    return True


class TestCovarianceMixedScaleAcceptance:
    """A well-conditioned matrix that mixes very large and very small variances -- exactly the
    shape a natural-scale ``(mu, s, upper, lower)`` covariance has when severity is measured in
    very large or very small units -- is accepted, and so is a uniformly tiny-scale one that
    a raw-eigenvalue rule would reject outright.
    """

    def test_mixed_scale_with_modest_correlation_is_accepted(self) -> None:
        # diag(1e12, 1e12, 1e-5, 1e-5) with a modest (0.3) correlation between every pair,
        # built from the correlation matrix directly so the off-diagonal entries are exactly
        # what a 0.3 correlation implies at these scales, not hand-computed by hand.
        diagonal = np.array([1e12, 1e12, 1e-5, 1e-5])
        correlation = 0.3 * np.ones((4, 4)) + 0.7 * np.eye(4)
        scale = np.sqrt(diagonal)
        matrix = correlation * np.outer(scale, scale)
        matrix = (matrix + matrix.T) / 2.0

        covariance = Covariance(
            names=("mu", "s", "upper", "lower"), matrix=tuple(tuple(row) for row in matrix)
        )

        assert covariance.matrix[0][0] == pytest.approx(1e12)
        assert covariance.matrix[2][2] == pytest.approx(1e-5)

    def test_uniformly_tiny_scale_identity_is_accepted(self) -> None:
        # 1e-14 * I: a raw-eigenvalue rule's tolerance would be 1e-12 * max(1, max|entry|) = 1e-12
        # here (since max|entry| = 1e-14 < 1), and the matrix's own eigenvalue is 1e-14 < 1e-12 --
        # that rule would reject this outright. The rule normalises to the correlation matrix
        # first, which is exactly the identity regardless of the common scale factor, so it is
        # accepted.
        matrix = 1e-14 * np.eye(3)

        covariance = Covariance(
            names=("mu", "s", "upper"), matrix=tuple(tuple(row) for row in matrix)
        )

        assert covariance.matrix[0][0] == pytest.approx(1e-14)


class TestCovarianceAcceptsEveryMatrixTheRawEigenvalueRuleAccepts:
    """Property: any finite matrix the raw-eigenvalue rule accepts, the units-independent rule
    also accepts.

    Matrices are built as ``A . A^T + eps*I`` (guaranteed positive semi-definite, nudged
    strictly positive-definite by ``eps*I``) and then rescaled per-parameter by a random
    power-of-ten diagonal congruence (``D . (A.A^T + eps*I) . D``, which preserves
    positive-definiteness exactly), so examples genuinely span the mixed-scale designs the fix
    targets, not just uniformly-scaled ones.

    The congruence transform is exactly symmetric, but a real ``Fit``'s covariance essentially
    never is, bit-for-bit, once floating-point round-off enters -- and ``Covariance`` itself
    tolerates that (``math.isclose``, ``rel_tol=1e-9``). Both :func:`numpy.linalg.eigvalsh` and
    ``models._is_positive_definite`` read only one triangle by default and ignore the other, so
    *averaging* the two triangles before computing eigenvalues is not a no-op and can turn a
    matrix the raw-eigenvalue rule accepts (reading its lower triangle) into one that is
    rejected. Each example therefore perturbs only the **upper** triangle by a small relative
    amount, safely inside ``Covariance``'s own symmetry tolerance, so the generated matrices
    are never bit-exactly symmetric and would catch that mistake.
    """

    @staticmethod
    @st.composite
    def _matrices(draw: st.DrawFn) -> np.ndarray:
        n = draw(st.integers(min_value=1, max_value=4))
        a_entries = draw(
            st.lists(
                st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
                min_size=n * n,
                max_size=n * n,
            )
        )
        a = np.array(a_entries, dtype=float).reshape(n, n)
        base = a @ a.T + 1e-6 * np.eye(n)

        exponents = draw(
            st.lists(
                st.integers(min_value=-8, max_value=8),
                min_size=n,
                max_size=n,
            )
        )
        d = np.diag([10.0**e for e in exponents])
        matrix = d @ base @ d

        # Perturb only the upper triangle (row < col), leaving the lower triangle at its
        # exact, congruence-transform-symmetric value -- within Covariance's rel_tol=1e-9
        # symmetry tolerance, but not bit-exactly symmetric, matching the shape of the
        # regression example below (`b` in the lower triangle, `b + 1e-10` in the upper).
        for i in range(n):
            for j in range(i + 1, n):
                relative_noise = draw(
                    st.floats(
                        min_value=-1e-10, max_value=1e-10, allow_nan=False, allow_infinity=False
                    )
                )
                matrix[i, j] = matrix[i, j] * (1.0 + relative_noise)
        return matrix

    @given(_matrices())
    @settings(max_examples=200, deadline=None)
    def test_covariance_accepts_every_matrix_the_raw_eigenvalue_rule_accepts(
        self, matrix: np.ndarray
    ) -> None:
        assume(bool(np.all(np.isfinite(matrix))))
        assume(_raw_eigenvalue_rule_accepts(matrix))

        assert _covariance_accepts(matrix)

    def test_asymmetric_noise_within_tolerance_that_the_raw_eigenvalue_rule_accepts_is_accepted(
        self,
    ) -> None:
        """A regression example: the raw-eigenvalue rule accepts this matrix (its lower triangle,
        ``b``, gives eigenvalues ``1 +- b`` = ``2 - 2e-12`` and ``2e-12``, both above the
        tolerance ``1e-12 * max(1, max|entry|) = 1e-12``), while a rule that averaged the two
        off-diagonal entries before computing eigenvalues would reject it (the averaged entry
        pushes the smaller eigenvalue negative). The rule must accept it.
        """
        b = 1.0 - 2e-12
        matrix = ((1.0, b + 1e-10), (b, 1.0))

        covariance = Covariance(names=("x", "y"), matrix=matrix)

        assert covariance.matrix == matrix


class TestCovarianceStillRejects:
    """The units-independent rule is not a rubber stamp: every
    matrix that was never a legitimate covariance -- singular, a non-positive variance, a
    perfectly (colinear) correlated pair, or a non-finite entry -- is still rejected.
    """

    def test_singular_matrix_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            Covariance(names=("mu", "s"), matrix=((0.0, 0.0), (0.0, 1.0)))

    def test_negative_diagonal_entry_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            Covariance(names=("mu", "s"), matrix=((-1.0, 0.0), (0.0, 1.0)))

    def test_perfect_correlation_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            Covariance(names=("mu", "s"), matrix=((1.0, 1.0), (1.0, 1.0)))

    def test_non_finite_entry_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            Covariance(names=("mu", "s"), matrix=((1.0, 0.0), (0.0, float("nan"))))


class TestFitDefaults:
    def test_defaults(self) -> None:
        fit = _ok_fit()

        assert fit.warnings == ()
        # One package-wide SCHEMA_VERSION, currently "1" -- Fit's own default tracks the same
        # constant, since report.py versions the whole card format, not each type independently.
        assert fit.schema_version == "1"
        assert fit.provenance == {}


class TestFitStatusParamsConsistency:
    def test_ok_status_with_no_params_raises(self) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params=None,
                covariance=None,
                log_likelihood=None,
                cells=_CELLS,
                status=Status.OK,
            )

    def test_ok_status_with_params_but_no_log_likelihood_raises(self) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params=_params(),
                covariance=_covariance(),
                log_likelihood=None,
                cells=_CELLS,
                status=Status.OK,
            )

    def test_non_ok_status_with_params_present_raises(self) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params=_params(),
                covariance=_covariance(),
                log_likelihood=-1.0,
                cells=_CELLS,
                status=Status.NOT_CONVERGED,
            )

    def test_non_ok_status_with_everything_absent_is_valid(self) -> None:
        fit = _fit_with_axis(_AXIS, status=Status.NOT_CONVERGED)

        assert fit.params is None
        assert fit.covariance is None
        assert fit.log_likelihood is None


class TestFitOutcomeAndDirection:
    def test_empty_outcome_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="",
                direction="decreasing",
                model="binomial",
                link="probit",
                params=_params(),
                covariance=_covariance(),
                log_likelihood=-1.0,
                cells=_CELLS,
                status=Status.OK,
            )

    def test_direction_outside_documented_values_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="sideways",
                model="binomial",
                link="probit",
                params=_params(),
                covariance=_covariance(),
                log_likelihood=-1.0,
                cells=_CELLS,
                status=Status.OK,
            )

    def test_increasing_direction_on_a_binomial_model_raises_value_error(self) -> None:
        """Binomial curves are decreasing only. ``Observations`` still accepts
        both directions (``tests/unit/test_types.py::TestDirectionAndOutcome``); this
        restriction is specific to ``Fit`` (and, transitively, ``Threshold``)."""
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="increasing",
                model="binomial",
                link="probit",
                params=_params(),
                covariance=_covariance(),
                log_likelihood=-1.0,
                cells=_CELLS,
                status=Status.OK,
            )


class TestFitModelAndLink:
    def test_binomial_requires_a_recognised_link(self) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="identity",
                params=_params(),
                covariance=_covariance(),
                log_likelihood=-1.0,
                cells=_CELLS,
                status=Status.OK,
            )

    @pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
    def test_binomial_accepts_each_documented_link(self, link: str) -> None:
        fit = Fit(
            axis=_AXIS,
            outcome="success",
            direction="decreasing",
            model="binomial",
            link=link,
            params=_params(),
            covariance=_covariance(),
            log_likelihood=-1.0,
            cells=_CELLS,
            status=Status.OK,
        )

        assert fit.link == link


class TestFitParams:
    def test_params_keys_must_be_exactly_mu_s_upper_lower(self) -> None:
        missing_lower = {k: v for k, v in _params().items() if k != "lower"}

        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params=missing_lower,
                covariance=_covariance(),
                log_likelihood=-1.0,
                cells=_CELLS,
                status=Status.OK,
            )

    def test_s_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params=_params(s=0.0),
                covariance=_covariance(),
                log_likelihood=-1.0,
                cells=_CELLS,
                status=Status.OK,
            )

    def test_lower_must_be_strictly_less_than_upper(self) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params=_params(upper=0.5, lower=0.5),
                covariance=_covariance(),
                log_likelihood=-1.0,
                cells=_CELLS,
                status=Status.OK,
            )

    def test_lower_below_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params=_params(lower=-0.1),
                covariance=_covariance(),
                log_likelihood=-1.0,
                cells=_CELLS,
                status=Status.OK,
            )

    def test_upper_above_one_raises(self) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params=_params(upper=1.1),
                covariance=_covariance(),
                log_likelihood=-1.0,
                cells=_CELLS,
                status=Status.OK,
            )


class TestFitCovariance:
    def test_covariance_names_must_equal_the_non_fixed_parameter_names(self) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params=_params(),
                covariance=_covariance(names=("mu", "s", "upper")),
                log_likelihood=-1.0,
                cells=_CELLS,
                status=Status.OK,
            )


class TestFitCells:
    def test_cells_must_not_be_empty(self) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params=_params(),
                covariance=_covariance(),
                log_likelihood=-1.0,
                cells=(),
                status=Status.OK,
            )


# ------------------------------------------------------------------------------------------
# ``Fit.cells`` must be strictly ascending in severity. Every exact-test rule in
# ``censoring.py`` reads ``cells`` in order -- RIGHT the last cell, LEFT the first nonzero cell,
# the fixed-sequence scans from each end -- so an unsorted or duplicated table made
# ``threshold()`` silently test the wrong levels. Nothing is sorted or pooled for the caller:
# malformed cells are an error. The check applies on every status, because the exact-test rules
# read ``cells`` of a *failed* fit (those are the fits they exist for).
# ------------------------------------------------------------------------------------------

# Every status a ``Fit`` may carry. UNREACHABLE and FAILS_AT_BASELINE describe a threshold and
# ``Fit`` rejects them for an unrelated reason (a ``Fit`` never carries a threshold-level status),
# so they cannot isolate the cells check.
_FIT_STATUSES: tuple[Status, ...] = (
    Status.OK,
    Status.SEPARATION,
    Status.NOT_CONVERGED,
    Status.CONTROL_INCOMPATIBLE,
)

# The zero-severity cell is always an all-success 100/100. A *failing* control under the fixed
# upper=1.0 of ``_params()`` on this log axis is rejected by a different rule, which would make an
# ordering test pass for the wrong reason.
_SEV_0 = Cell(severity=0.0, successes=100, trials=100)
_SEV_1 = Cell(severity=1.0, successes=90, trials=100)
_SEV_1_AGAIN = Cell(severity=1.0, successes=80, trials=100)
_SEV_2 = Cell(severity=2.0, successes=60, trials=100)
_SEV_3 = Cell(severity=3.0, successes=30, trials=100)

# Each shape breaks "strictly increasing" in a different place, so a check that compared only the
# first pair, or only the last, or only the ends, would miss one of them.
_UNSORTED: dict[str, tuple[Cell, ...]] = {
    "two-cells-descending": (_SEV_1, _SEV_0),
    "fully-reversed": (_SEV_3, _SEV_2, _SEV_1, _SEV_0),
    "swap-in-the-middle": (_SEV_0, _SEV_2, _SEV_1, _SEV_3),
    "swap-at-the-end": (_SEV_0, _SEV_1, _SEV_3, _SEV_2),
    "returns-to-an-earlier-severity": (_SEV_0, _SEV_2, _SEV_3, _SEV_1),
    "control-last": (_SEV_1, _SEV_2, _SEV_0),
}
_DUPLICATED: dict[str, tuple[Cell, ...]] = {
    "control-repeated": (_SEV_0, _SEV_0, _SEV_1),
    "duplicate-in-the-middle": (_SEV_0, _SEV_1, _SEV_1_AGAIN, _SEV_2),
    "duplicate-at-the-end": (_SEV_0, _SEV_1, _SEV_1_AGAIN),
    "exact-repeat-of-a-row": (_SEV_0, _SEV_1, _SEV_1),
    "only-two-cells-at-one-severity": (_SEV_1, _SEV_1_AGAIN),
}


class TestFitCellsStrictlyAscending:
    @pytest.mark.parametrize("status", _FIT_STATUSES, ids=lambda s: s.name)
    @pytest.mark.parametrize("cells", _UNSORTED.values(), ids=list(_UNSORTED))
    def test_cells_not_sorted_ascending_raise_value_error(
        self, cells: tuple[Cell, ...], status: Status
    ) -> None:
        with pytest.raises(ValueError):
            _fit_with_axis(_AXIS, status=status, cells=cells)

    @pytest.mark.parametrize("status", _FIT_STATUSES, ids=lambda s: s.name)
    @pytest.mark.parametrize("cells", _DUPLICATED.values(), ids=list(_DUPLICATED))
    def test_two_cells_at_the_same_severity_raise_value_error(
        self, cells: tuple[Cell, ...], status: Status
    ) -> None:
        with pytest.raises(ValueError):
            _fit_with_axis(_AXIS, status=status, cells=cells)

    @pytest.mark.parametrize("cells", _UNSORTED.values(), ids=list(_UNSORTED))
    def test_the_same_cells_in_ascending_order_construct(self, cells: tuple[Cell, ...]) -> None:
        """What each rejected shape above lacks is order and nothing else: sorted, the identical
        counts build a fit, and it holds them in that order."""
        ascending = tuple(sorted(cells, key=lambda c: c.severity))

        fit = _fit_with_axis(_AXIS, cells=ascending)

        assert fit.cells == ascending

    @pytest.mark.parametrize("status", _FIT_STATUSES, ids=lambda s: s.name)
    def test_strictly_ascending_cells_construct_in_the_order_given(self, status: Status) -> None:
        cells = (_SEV_0, _SEV_1, _SEV_2, _SEV_3)

        fit = _fit_with_axis(_AXIS, status=status, cells=cells)

        assert fit.cells == cells

    @pytest.mark.parametrize("status", _FIT_STATUSES, ids=lambda s: s.name)
    def test_a_single_cell_is_accepted(self, status: Status) -> None:
        fit = _fit_with_axis(_AXIS, status=status, cells=(_SEV_1,))

        assert fit.cells == (_SEV_1,)

    def test_unsorted_cells_given_as_a_list_are_rejected(self) -> None:
        """``Fit`` normalises ``cells`` to a tuple before validating; a list must not slip past
        the order check by being a different sequence type."""
        with pytest.raises(ValueError):
            _fit_with_axis(_AXIS, cells=cast(Any, [_SEV_1, _SEV_0]))

    def test_replacing_cells_on_a_valid_fit_runs_the_same_check(self) -> None:
        """``dataclasses.replace`` re-runs ``__post_init__``, so a valid fit cannot be turned
        into an unsorted one that way."""
        fit = _ok_fit()

        with pytest.raises(ValueError):
            dataclasses.replace(fit, cells=(_SEV_1, _SEV_0))


def _ordering_error(cells: tuple[Cell, ...], status: Status = Status.OK) -> str:
    with pytest.raises(ValueError) as exc_info:
        _fit_with_axis(_AXIS, status=status, cells=cells)
    return str(exc_info.value)


class TestFitCellsOrderingMessage:
    """The message names the first out-of-order pair. Substrings only: the
    exact wording is not API. Severities are chosen so that their text is unambiguous (no one
    severity's text is a substring of another's), and that it cites no private document is
    checked with the other user-visible messages in
    ``test_warning_strings_avoid_private_doc_citations.py``."""

    _LOW = Cell(severity=0.0, successes=100, trials=100)
    _A = Cell(severity=1.5, successes=90, trials=100)
    _A_AGAIN = Cell(severity=1.5, successes=80, trials=100)
    _B = Cell(severity=7.25, successes=60, trials=100)
    _C = Cell(severity=3.125, successes=50, trials=100)
    _D = Cell(severity=9.5, successes=30, trials=100)
    _E = Cell(severity=6.0625, successes=20, trials=100)

    def test_descending_pair_message_names_both_severities(self) -> None:
        message = _ordering_error((self._A, self._LOW))

        assert "1.5" in message
        assert "0.0" in message

    def test_message_names_the_first_descending_pair_and_not_a_later_one(self) -> None:
        """Order is ``0.0, 1.5, 7.25, 3.125, 9.5, 6.0625``: the pairs ``(7.25, 3.125)`` and
        ``(9.5, 6.0625)`` are both out of order, and the first is the one reported."""
        message = _ordering_error((self._LOW, self._A, self._B, self._C, self._D, self._E))

        assert "7.25" in message
        assert "3.125" in message
        assert "9.5" not in message
        assert "6.0625" not in message

    def test_message_for_duplicate_severities_names_the_repeated_severity(self) -> None:
        message = _ordering_error((self._LOW, self._A, self._A_AGAIN))

        assert "1.5" in message

    def test_message_names_a_leading_duplicate_not_a_later_descent(self) -> None:
        """``0.0, 1.5, 1.5, 7.25, 3.125``: the duplicate at ``1.5`` comes first and is the one
        named, not the later ``(7.25, 3.125)`` descent."""
        message = _ordering_error((self._LOW, self._A, self._A_AGAIN, self._B, self._C))

        assert "1.5" in message
        assert "7.25" not in message
        assert "3.125" not in message

    @pytest.mark.parametrize("status", _FIT_STATUSES, ids=lambda s: s.name)
    def test_message_names_the_pair_on_every_status(self, status: Status) -> None:
        message = _ordering_error((self._B, self._C), status=status)

        assert "7.25" in message
        assert "3.125" in message

    def test_message_names_the_field(self) -> None:
        message = _ordering_error((self._B, self._C))

        assert "cells" in message


class TestFitIsFrozen:
    def test_is_frozen(self) -> None:
        fit = _ok_fit()

        with pytest.raises(dataclasses.FrozenInstanceError):
            cast(Any, fit).status = Status.NOT_CONVERGED


class TestFitRequiresCovarianceWhenOk:
    """``status is OK`` requires ``covariance`` to be present. Unlike the "covariance.names must
    equal the non-fixed parameter names" check (which only fires when ``covariance`` is not
    ``None``), this one fires when it is absent."""

    def test_ok_status_without_covariance_raises(self) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params=_params(),
                covariance=None,
                log_likelihood=-1.0,
                cells=_CELLS,
                status=Status.OK,
            )


class TestFitStatusExcludesThresholdLevelStatuses:
    """``UNREACHABLE`` and ``FAILS_AT_BASELINE`` describe whether a *threshold* can be
    solved for given a successful fit, not whether the fit itself succeeded, so ``Fit.status``
    may never be either."""

    @pytest.mark.parametrize("status", [Status.UNREACHABLE, Status.FAILS_AT_BASELINE])
    def test_threshold_level_status_on_a_fit_raises(self, status: Status) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params=None,
                covariance=None,
                log_likelihood=None,
                cells=_CELLS,
                status=status,
            )


class TestFitControlIncompatibleLikelihood:
    """On a log axis, fixing ``upper=1.0`` while the zero-severity
    control has any failures drives that cell's likelihood to zero -- ``CONTROL_INCOMPATIBLE``
    territory, not a fittable ``OK`` result. A linear axis has no ``log(0)`` special case, so
    the same counts are unproblematic there.
    """

    _FAILING_CONTROL_CELLS = (
        Cell(severity=0.0, successes=90, trials=100),
        Cell(severity=1.0, successes=40, trials=100),
    )

    def test_fixed_upper_one_with_a_failing_control_on_a_log_axis_raises(self) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params=_params(upper=1.0),
                covariance=_covariance(),
                log_likelihood=-1.0,
                cells=self._FAILING_CONTROL_CELLS,
                status=Status.OK,
            )

    def test_the_same_counts_on_a_linear_axis_are_accepted(self) -> None:
        linear_axis = Axis(name="axis_2", unit="m/s", scale="linear")
        fit = Fit(
            axis=linear_axis,
            outcome="success",
            direction="decreasing",
            model="binomial",
            link="probit",
            params=_params(upper=1.0),
            covariance=_covariance(),
            log_likelihood=-1.0,
            cells=self._FAILING_CONTROL_CELLS,
            status=Status.OK,
        )

        assert fit.status is Status.OK


class TestFitClusterIds:
    """``Fit.cluster_ids`` records which cluster each fitted cell's data came
    from, so a method assuming independence can check for overlap. ``None``
    (the default) means no cluster information was recorded, not "no clustering" -- unlike
    ``Observations.cluster``, which stores one id per row, this is coarser: one id set for the
    whole fit."""

    def test_duplicate_cluster_id_raises(self) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params=_params(),
                covariance=_covariance(),
                log_likelihood=-1.0,
                cells=_CELLS,
                status=Status.OK,
                cluster_ids=("a", "a"),
            )

    def test_bool_cluster_id_raises(self) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params=_params(),
                covariance=_covariance(),
                log_likelihood=-1.0,
                cells=_CELLS,
                status=Status.OK,
                cluster_ids=("a", True),
            )

    def test_none_cluster_id_raises(self) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params=_params(),
                covariance=_covariance(),
                log_likelihood=-1.0,
                cells=_CELLS,
                status=Status.OK,
                cluster_ids=("a", None),
            )

    def test_empty_cluster_ids_raises(self) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params=_params(),
                covariance=_covariance(),
                log_likelihood=-1.0,
                cells=_CELLS,
                status=Status.OK,
                cluster_ids=(),
            )

    def test_unique_cluster_ids_are_accepted(self) -> None:
        fit = Fit(
            axis=_AXIS,
            outcome="success",
            direction="decreasing",
            model="binomial",
            link="probit",
            params=_params(),
            covariance=_covariance(),
            log_likelihood=-1.0,
            cells=_CELLS,
            status=Status.OK,
            cluster_ids=("a", "b"),
        )

        assert fit.cluster_ids == ("a", "b")

    def test_default_is_none(self) -> None:
        fit = _ok_fit()

        assert fit.cluster_ids is None


class TestThresholdTerminalStatuses:
    """``UNREACHABLE``, ``CONTROL_INCOMPATIBLE`` and ``FAILS_AT_BASELINE``: no threshold at
    all, ``censoring`` is always ``NONE``."""

    @pytest.mark.parametrize(
        "status", [Status.UNREACHABLE, Status.CONTROL_INCOMPATIBLE, Status.FAILS_AT_BASELINE]
    )
    def test_valid_construction_has_no_value_lo_or_hi(self, status: Status) -> None:
        threshold = _threshold(
            value=None,
            lo=None,
            hi=None,
            interval_method="exact_bound",
            censoring=Censoring.NONE,
            status=status,
        )

        assert threshold.value is None
        assert threshold.lo is None
        assert threshold.hi is None
        assert threshold.censoring is Censoring.NONE

    @pytest.mark.parametrize(
        "status", [Status.UNREACHABLE, Status.CONTROL_INCOMPATIBLE, Status.FAILS_AT_BASELINE]
    )
    def test_a_value_present_raises(self, status: Status) -> None:
        with pytest.raises(ValueError):
            _threshold(value=0.05, lo=None, hi=None, interval_method="exact_bound", status=status)

    @pytest.mark.parametrize(
        "status", [Status.UNREACHABLE, Status.CONTROL_INCOMPATIBLE, Status.FAILS_AT_BASELINE]
    )
    def test_censoring_other_than_none_raises(self, status: Status) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=None,
                lo=10.0,
                hi=None,
                interval_method="exact_bound",
                censoring=Censoring.RIGHT,
                status=status,
            )


class TestThresholdSeparationAndNotConverged:
    @pytest.mark.parametrize("status", [Status.SEPARATION, Status.NOT_CONVERGED])
    @pytest.mark.parametrize("censoring", [Censoring.NONE, Censoring.RIGHT, Censoring.LEFT])
    def test_valid_construction_has_no_value_and_an_exact_bound_interval_method(
        self, status: Status, censoring: Censoring
    ) -> None:
        lo = 0.02 if censoring in (Censoring.NONE, Censoring.RIGHT) else None
        hi = 0.09 if censoring in (Censoring.NONE, Censoring.LEFT) else None

        threshold = _threshold(
            value=None,
            lo=lo,
            hi=hi,
            interval_method="exact_bound",
            censoring=censoring,
            status=status,
        )

        assert threshold.value is None
        assert threshold.interval_method == "exact_bound"

    @pytest.mark.parametrize("status", [Status.SEPARATION, Status.NOT_CONVERGED])
    def test_a_value_present_raises(self, status: Status) -> None:
        with pytest.raises(ValueError):
            _threshold(value=0.05, lo=0.01, hi=0.1, interval_method="exact_bound", status=status)

    @pytest.mark.parametrize("status", [Status.SEPARATION, Status.NOT_CONVERGED])
    def test_open_upper_censoring_is_rejected(self, status: Status) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=None,
                lo=0.01,
                hi=None,
                interval_method="exact_bound",
                censoring=Censoring.OPEN_UPPER,
                status=status,
            )

    @pytest.mark.parametrize("status", [Status.SEPARATION, Status.NOT_CONVERGED])
    def test_interval_method_other_than_exact_bound_raises(self, status: Status) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=None,
                lo=0.02,
                hi=0.09,
                interval_method="profile",
                censoring=Censoring.NONE,
                status=status,
            )


class TestThresholdRightCensoring:
    def test_valid_construction(self) -> None:
        threshold = _threshold(
            value=None,
            lo=10.0,
            hi=None,
            interval_method="exact_bound",
            censoring=Censoring.RIGHT,
            status=Status.OK,
        )

        assert threshold.value is None
        assert threshold.lo == 10.0
        assert threshold.hi is None

    def test_a_value_present_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=0.05,
                lo=10.0,
                hi=None,
                interval_method="exact_bound",
                censoring=Censoring.RIGHT,
                status=Status.OK,
            )

    def test_missing_lo_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=None,
                lo=None,
                hi=None,
                interval_method="exact_bound",
                censoring=Censoring.RIGHT,
                status=Status.OK,
            )

    def test_hi_present_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=None,
                lo=10.0,
                hi=12.0,
                interval_method="exact_bound",
                censoring=Censoring.RIGHT,
                status=Status.OK,
            )

    def test_interval_method_other_than_exact_bound_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=None,
                lo=10.0,
                hi=None,
                interval_method="profile",
                censoring=Censoring.RIGHT,
                status=Status.OK,
            )


class TestThresholdLeftCensoring:
    def test_valid_construction(self) -> None:
        threshold = _threshold(
            value=None,
            lo=None,
            hi=0.01,
            interval_method="exact_bound",
            censoring=Censoring.LEFT,
            status=Status.OK,
        )

        assert threshold.value is None
        assert threshold.hi == 0.01
        assert threshold.lo is None

    def test_a_value_present_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=0.01,
                lo=None,
                hi=0.01,
                interval_method="exact_bound",
                censoring=Censoring.LEFT,
                status=Status.OK,
            )

    def test_missing_hi_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=None,
                lo=None,
                hi=None,
                interval_method="exact_bound",
                censoring=Censoring.LEFT,
                status=Status.OK,
            )

    def test_lo_present_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=None,
                lo=0.001,
                hi=0.01,
                interval_method="exact_bound",
                censoring=Censoring.LEFT,
                status=Status.OK,
            )

    def test_interval_method_other_than_exact_bound_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=None,
                lo=None,
                hi=0.01,
                interval_method="profile",
                censoring=Censoring.LEFT,
                status=Status.OK,
            )


class TestThresholdOpenCensoring:
    def test_open_upper_valid_construction(self) -> None:
        threshold = _threshold(
            value=0.05,
            lo=0.03,
            hi=None,
            interval_method="profile",
            censoring=Censoring.OPEN_UPPER,
            status=Status.OK,
        )

        assert threshold.value == 0.05
        assert threshold.lo == 0.03
        assert threshold.hi is None

    def test_open_upper_requires_status_ok(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=0.05,
                lo=0.03,
                hi=None,
                interval_method="profile",
                censoring=Censoring.OPEN_UPPER,
                status=Status.NOT_CONVERGED,
            )

    def test_open_lower_valid_construction(self) -> None:
        threshold = _threshold(
            value=0.05,
            lo=None,
            hi=0.08,
            interval_method="profile",
            censoring=Censoring.OPEN_LOWER,
            status=Status.OK,
        )

        assert threshold.value == 0.05
        assert threshold.hi == 0.08
        assert threshold.lo is None

    def test_open_lower_requires_status_ok(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=0.05,
                lo=None,
                hi=0.08,
                interval_method="profile",
                censoring=Censoring.OPEN_LOWER,
                status=Status.SEPARATION,
            )


class TestThresholdNoneCensoringWithOkStatus:
    def test_valid_construction(self) -> None:
        threshold = _threshold(
            value=0.05,
            lo=0.03,
            hi=0.08,
            interval_method="profile",
            censoring=Censoring.NONE,
            status=Status.OK,
        )

        assert threshold.lo is not None
        assert threshold.value is not None
        assert threshold.hi is not None
        assert threshold.lo <= threshold.value <= threshold.hi

    def test_missing_value_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=None,
                lo=0.03,
                hi=0.08,
                interval_method="profile",
                censoring=Censoring.NONE,
                status=Status.OK,
            )

    def test_value_outside_lo_hi_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=0.5,
                lo=0.03,
                hi=0.08,
                interval_method="profile",
                censoring=Censoring.NONE,
                status=Status.OK,
            )


class TestThresholdFloatsAreFiniteAndNonNegative:
    def test_negative_value_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=-0.1,
                lo=-0.2,
                hi=0.0,
                interval_method="profile",
                censoring=Censoring.NONE,
                status=Status.OK,
            )

    def test_non_finite_value_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=float("inf"),
                lo=0.0,
                hi=1.0,
                interval_method="profile",
                censoring=Censoring.NONE,
                status=Status.OK,
            )

    def test_nan_lo_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=0.5,
                lo=float("nan"),
                hi=1.0,
                interval_method="profile",
                censoring=Censoring.NONE,
                status=Status.OK,
            )


class TestThresholdLevel:
    @pytest.mark.parametrize("level", [0.0, 1.0])
    def test_level_at_the_endpoints_raises(self, level: float) -> None:
        with pytest.raises(ValueError):
            _threshold(value=0.05, lo=0.03, hi=0.08, level=level, interval_method="profile")


class TestThresholdAxisMatchesFitAxis:
    def test_mismatched_axis_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=0.05,
                lo=0.03,
                hi=0.08,
                interval_method="profile",
                axis=_OTHER_NAME_AXIS,
                fit=_ok_fit(),
            )


class TestThresholdDirectionMatchesFitDirection:
    def test_matching_direction_is_accepted(self) -> None:
        threshold = _threshold(value=0.05, lo=0.03, hi=0.08, interval_method="profile")

        assert threshold.direction == threshold.fit.direction

    def test_increasing_direction_on_the_underlying_fit_is_rejected(self) -> None:
        """A binomial ``Fit`` can never have ``direction="increasing"``, so a
        ``Threshold.direction`` != ``Fit.direction`` mismatch (a ``"decreasing"`` threshold on an
        ``"increasing"`` fit) cannot even be constructed for the binomial-only model: there is
        no second valid direction value to mismatch against, since neither side may
        legitimately be ``"increasing"``. Both ``direction`` fields are set to ``"increasing"``
        here (matching each other, so the mismatch check alone would not fire) to isolate the
        direction rejection.
        """
        with pytest.raises(ValueError):
            _threshold(
                value=0.05,
                lo=0.03,
                hi=0.08,
                interval_method="profile",
                direction="increasing",
                fit=_fit_with_axis(_AXIS, direction="increasing"),
            )


class TestThresholdIntervalMethod:
    def test_unknown_interval_method_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(value=0.05, lo=0.03, hi=0.08, interval_method="bootstrap")

    @pytest.mark.parametrize("method", ["profile", "delta"])
    def test_documented_interval_methods_are_accepted_for_a_closed_interval(
        self, method: str
    ) -> None:
        threshold = _threshold(value=0.05, lo=0.03, hi=0.08, interval_method=method)

        assert threshold.interval_method == method


class TestThresholdDefaultsAndFrozen:
    def test_defaults(self) -> None:
        threshold = _threshold(value=0.05, lo=0.03, hi=0.08, interval_method="profile")

        assert threshold.warnings == ()
        # See TestFitDefaults's own comment -- one package-wide SCHEMA_VERSION, currently "1".
        assert threshold.schema_version == "1"
        assert threshold.provenance == {}

    def test_is_frozen(self) -> None:
        threshold = _threshold(value=0.05, lo=0.03, hi=0.08, interval_method="profile")

        with pytest.raises(dataclasses.FrozenInstanceError):
            cast(Any, threshold).value = 1.0


class TestThresholdFitStatusConsistency:
    """A threshold's own ``status``/``censoring`` constrain what ``fit.status`` may be. First, a
    ``value``, or ``status`` OK, or censoring OPEN_UPPER/OPEN_LOWER, requires
    ``fit.status is OK``; second, ``status`` SEPARATION/NOT_CONVERGED/CONTROL_INCOMPATIBLE
    requires ``fit.status`` to match. Both are exercised as rejections, each against a
    deliberately mismatched ``fit=``.

    ``FAILS_AT_BASELINE`` places no constraint on the fit status, and is exercised as an accept
    test. ``UNREACHABLE`` places a narrower one: it is accepted on a fit that is ``OK``,
    ``SEPARATION`` or ``NOT_CONVERGED``, and rejected on a ``CONTROL_INCOMPATIBLE`` fit, which is
    reported as itself before any target is examined. ``UNREACHABLE`` is reported for a target
    outside ``(0, 1)`` on a fit of any other status (no binomial curve crosses such a target, so
    no fitted curve is needed to say so), as well as for a target the fitted asymptotes cannot
    reach on a converged one. Of the fourteen allowed (status, censoring) combinations these two
    are the ones that are not tied to a single fit status by the first two rules.
    """

    def test_a_value_with_a_mismatched_fit_status_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=0.05,
                lo=0.03,
                hi=0.08,
                interval_method="profile",
                censoring=Censoring.NONE,
                status=Status.OK,
                fit=_fit_with_axis(_AXIS, status=Status.SEPARATION),
            )

    @pytest.mark.parametrize(
        "fit_status", [Status.OK, Status.SEPARATION, Status.NOT_CONVERGED], ids=lambda s: s.name
    )
    def test_unreachable_accepts_a_fit_of_any_status_a_threshold_can_be_built_on(
        self, fit_status: Status
    ) -> None:
        threshold = _threshold(
            value=None,
            lo=None,
            hi=None,
            interval_method="exact_bound",
            censoring=Censoring.NONE,
            status=Status.UNREACHABLE,
            fit=_fit_with_axis(_AXIS, status=fit_status),
        )

        assert threshold.fit.status is fit_status
        assert threshold.status is Status.UNREACHABLE

    def test_unreachable_with_a_control_incompatible_fit_raises(self) -> None:
        """A ``CONTROL_INCOMPATIBLE`` fit is reported as itself before any target is examined, so
        a threshold that says ``UNREACHABLE`` on top of one has misdescribed the fit. The match
        pins the rejection to this rule rather than to some other ``ValueError``."""
        with pytest.raises(ValueError, match=r"(?s)UNREACHABLE.*CONTROL_INCOMPATIBLE"):
            _threshold(
                value=None,
                lo=None,
                hi=None,
                interval_method="exact_bound",
                censoring=Censoring.NONE,
                status=Status.UNREACHABLE,
                fit=_fit_with_axis(_AXIS, status=Status.CONTROL_INCOMPATIBLE),
            )

    @pytest.mark.parametrize(
        ("value", "lo", "hi", "censoring", "interval_method"),
        [
            (0.05, None, None, Censoring.NONE, "exact_bound"),
            (None, 0.03, None, Censoring.NONE, "exact_bound"),
            (None, None, 0.08, Censoring.NONE, "exact_bound"),
            (None, None, None, Censoring.RIGHT, "exact_bound"),
            (None, None, None, Censoring.NONE, "profile"),
        ],
        ids=["value", "lo", "hi", "censoring", "method"],
    )
    def test_unreachable_on_a_failed_fit_still_carries_no_numbers(
        self,
        value: float | None,
        lo: float | None,
        hi: float | None,
        censoring: Censoring,
        interval_method: str,
    ) -> None:
        """Loosening which fit it may sit on does not loosen what it carries: no value, no
        bounds, no censoring, and ``exact_bound`` as the method."""
        with pytest.raises(ValueError):
            _threshold(
                value=value,
                lo=lo,
                hi=hi,
                interval_method=interval_method,
                censoring=censoring,
                status=Status.UNREACHABLE,
                fit=_fit_with_axis(_AXIS, status=Status.SEPARATION),
            )

    def test_open_upper_with_a_non_ok_fit_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=0.05,
                lo=0.03,
                hi=None,
                interval_method="profile",
                censoring=Censoring.OPEN_UPPER,
                status=Status.OK,
                fit=_fit_with_axis(_AXIS, status=Status.SEPARATION),
            )

    def test_separation_status_with_a_mismatched_fit_status_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=None,
                lo=0.02,
                hi=0.09,
                interval_method="exact_bound",
                censoring=Censoring.NONE,
                status=Status.SEPARATION,
                fit=_fit_with_axis(_AXIS, status=Status.NOT_CONVERGED),
            )

    def test_control_incompatible_status_with_an_ok_fit_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=None,
                lo=None,
                hi=None,
                interval_method="exact_bound",
                censoring=Censoring.NONE,
                status=Status.CONTROL_INCOMPATIBLE,
                fit=_fit_with_axis(_AXIS, status=Status.OK),
            )

    @pytest.mark.parametrize(
        "fit_status",
        [Status.OK, Status.SEPARATION, Status.NOT_CONVERGED, Status.CONTROL_INCOMPATIBLE],
    )
    def test_fails_at_baseline_accepts_a_fit_of_any_status(self, fit_status: Status) -> None:
        threshold = _threshold(
            value=None,
            lo=None,
            hi=None,
            interval_method="exact_bound",
            censoring=Censoring.NONE,
            status=Status.FAILS_AT_BASELINE,
            fit=_fit_with_axis(_AXIS, status=fit_status),
        )

        assert threshold.fit.status is fit_status


class TestThresholdLoHiOrdering:
    """When both ``lo`` and ``hi`` are set, ``SEPARATION``/``NOT_CONVERGED`` exact-bound
    brackets require ``lo < hi`` strictly; every other case (the ``OK``+``NONE`` closed
    interval) only requires ``lo <= hi``."""

    def test_separation_bracket_with_equal_lo_and_hi_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=None,
                lo=0.05,
                hi=0.05,
                interval_method="exact_bound",
                censoring=Censoring.NONE,
                status=Status.SEPARATION,
            )

    def test_not_converged_bracket_with_lo_greater_than_hi_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=None,
                lo=0.09,
                hi=0.02,
                interval_method="exact_bound",
                censoring=Censoring.NONE,
                status=Status.NOT_CONVERGED,
            )

    def test_ok_none_with_equal_lo_and_hi_is_accepted(self) -> None:
        threshold = _threshold(
            value=0.05,
            lo=0.05,
            hi=0.05,
            interval_method="profile",
            censoring=Censoring.NONE,
            status=Status.OK,
        )

        assert threshold.lo == threshold.hi == threshold.value


class TestThresholdLogAxisPositivity:
    """On a log axis, ``value``/``lo``/``hi`` must each be strictly positive when
    present -- ``0.0`` is not a reachable severity on a log scale. A linear axis keeps the
    existing ``>= 0`` rule."""

    def test_zero_value_on_log_axis_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=0.0,
                lo=0.0,
                hi=0.08,
                interval_method="profile",
                censoring=Censoring.NONE,
                status=Status.OK,
            )

    def test_zero_lo_on_log_axis_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=0.05,
                lo=0.0,
                hi=0.08,
                interval_method="profile",
                censoring=Censoring.NONE,
                status=Status.OK,
            )

    def test_zero_hi_on_log_axis_raises(self) -> None:
        """Uses LEFT censoring (only ``hi`` set, no ``lo``/``value`` to order against it) so
        this isolates the log-axis positivity rule alone: with ``value``/``lo`` also present,
        ``hi=0.0`` would already violate the ``lo <= value <= hi`` ordering rule, which would
        mask whether the positivity rule is implemented at all."""
        with pytest.raises(ValueError):
            _threshold(
                value=None,
                lo=None,
                hi=0.0,
                interval_method="exact_bound",
                censoring=Censoring.LEFT,
                status=Status.OK,
            )

    def test_zero_value_on_linear_axis_is_accepted(self) -> None:
        linear_axis = Axis(name="axis_2", unit="m/s", scale="linear")

        threshold = _threshold(
            value=0.0,
            lo=0.0,
            hi=1.0,
            interval_method="profile",
            censoring=Censoring.NONE,
            status=Status.OK,
            axis=linear_axis,
        )

        assert threshold.value == 0.0


class TestThresholdIntervalMethodByStatusAndCensoring:
    """``OPEN_UPPER``/``OPEN_LOWER`` require ``"profile"`` specifically (not just any
    valid ``interval_method``), and ``status`` OK with censoring ``NONE`` requires ``"profile"``
    or ``"delta"`` -- ``"exact_bound"`` is for the censored/non-convergent fallback paths only.
    """

    def test_open_upper_with_delta_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=0.05,
                lo=0.03,
                hi=None,
                interval_method="delta",
                censoring=Censoring.OPEN_UPPER,
                status=Status.OK,
            )

    def test_open_lower_with_delta_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=0.05,
                lo=None,
                hi=0.08,
                interval_method="delta",
                censoring=Censoring.OPEN_LOWER,
                status=Status.OK,
            )

    def test_ok_none_with_exact_bound_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=0.05,
                lo=0.03,
                hi=0.08,
                interval_method="exact_bound",
                censoring=Censoring.NONE,
                status=Status.OK,
            )


class TestThresholdDependence:
    """``Threshold.dependence`` is required, with no default -- a wrong default (like a wrong
    default ``definition``) would silently mislabel a clustered result as independent. Only
    ``"independent"`` is valid in v1; ``"paired"`` is not implemented."""

    def test_dependence_other_than_independent_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=0.05,
                lo=0.03,
                hi=0.08,
                interval_method="profile",
                dependence="paired",
            )

    def test_dependence_is_required_with_no_default(self) -> None:
        with pytest.raises(TypeError):
            cast(Any, Threshold)(
                axis=_AXIS,
                definition=Definition.absolute(0.95),
                direction="decreasing",
                value=0.05,
                lo=0.03,
                hi=0.08,
                level=0.95,
                interval_method="profile",
                censoring=Censoring.NONE,
                status=Status.OK,
                fit=_ok_fit(),
                # dependence intentionally omitted
            )

    def test_independent_is_accepted(self) -> None:
        threshold = _threshold(value=0.05, lo=0.03, hi=0.08, interval_method="profile")

        assert threshold.dependence == "independent"


class TestThresholdHasNoNPerLevelField:
    """``n_per_level`` is not a field. Per-level n is read from
    ``threshold.fit.cells`` instead (each :class:`~robustmargin.Cell` already carries its own
    ``trials``), so a second, separately-populated copy is not part of the contract."""

    def test_n_per_level_is_not_a_field(self) -> None:
        field_names = {f.name for f in dataclasses.fields(Threshold)}

        assert "n_per_level" not in field_names


class TestRatioAxisGuard:
    def test_mismatched_axis_name_raises(self) -> None:
        with pytest.raises(ValueError):
            _ratio(
                estimate=1.0,
                lo=0.5,
                hi=2.0,
                shape=IntervalShape.BOUNDED,
                threshold_b=_threshold(
                    value=0.1,
                    lo=0.06,
                    hi=0.16,
                    interval_method="profile",
                    axis=_OTHER_NAME_AXIS,
                ),
            )

    def test_mismatched_unit_raises(self) -> None:
        with pytest.raises(ValueError):
            _ratio(
                estimate=1.0,
                lo=0.5,
                hi=2.0,
                shape=IntervalShape.BOUNDED,
                threshold_b=_threshold(
                    value=0.1,
                    lo=0.06,
                    hi=0.16,
                    interval_method="profile",
                    axis=_OTHER_UNIT_AXIS,
                ),
            )

    def test_matching_axis_name_and_unit_is_accepted(self) -> None:
        ratio = _ratio(estimate=1.0, lo=0.5, hi=2.0, shape=IntervalShape.BOUNDED)

        assert ratio.threshold_a.axis == ratio.threshold_b.axis


class TestRatioEstimateNoneWhenNotOkOrCensored:
    def test_status_not_ok_forbids_an_estimate(self) -> None:
        with pytest.raises(ValueError):
            _ratio(
                estimate=1.0,
                lo=None,
                hi=None,
                shape=IntervalShape.UNBOUNDED,
                status=Status.NOT_CONVERGED,
            )

    def test_censoring_other_than_none_forbids_an_estimate(self) -> None:
        with pytest.raises(ValueError):
            _ratio(
                estimate=1.0,
                lo=None,
                hi=None,
                shape=IntervalShape.UNBOUNDED,
                censoring=Censoring.RIGHT,
            )

    def test_status_not_ok_with_no_estimate_or_shape_is_valid(self) -> None:
        """A failed or censored ratio carries no numbers at all -- ``shape`` included, not just
        ``estimate``. ``shape=None`` is the valid encoding here; a shape set on such a ratio is
        rejected by ``test_status_not_ok_with_a_shape_set_raises`` below."""
        ratio = _ratio(
            estimate=None,
            lo=None,
            hi=None,
            shape=None,
            status=Status.NOT_CONVERGED,
        )

        assert ratio.estimate is None
        assert ratio.shape is None

    def test_status_not_ok_with_a_shape_set_raises(self) -> None:
        with pytest.raises(ValueError):
            _ratio(
                estimate=None,
                lo=None,
                hi=None,
                shape=IntervalShape.UNBOUNDED,
                status=Status.NOT_CONVERGED,
            )

    def test_censoring_other_than_none_with_lo_set_raises(self) -> None:
        with pytest.raises(ValueError):
            _ratio(
                estimate=None,
                lo=0.5,
                hi=None,
                shape=None,
                status=Status.OK,
                censoring=Censoring.RIGHT,
            )

    def test_censoring_other_than_none_with_a_shape_set_raises(self) -> None:
        with pytest.raises(ValueError):
            _ratio(
                estimate=None,
                lo=None,
                hi=None,
                shape=IntervalShape.BOUNDED,
                status=Status.OK,
                censoring=Censoring.RIGHT,
            )

    def test_ok_and_none_without_a_shape_raises(self) -> None:
        """The mirror rule: status OK with censoring NONE *requires* a shape."""
        with pytest.raises(ValueError):
            _ratio(estimate=1.0, lo=0.5, hi=2.0, shape=None, status=Status.OK)


class TestRatioShapeBounded:
    def test_valid_construction(self) -> None:
        # estimate=1.0 matches _ratio()'s default threshold_a/threshold_b ratio (0.05 / 0.05),
        # as the estimate must equal threshold_a.value / threshold_b.value.
        ratio = _ratio(estimate=1.0, lo=1.0, hi=2.0, shape=IntervalShape.BOUNDED)

        assert ratio.lo is not None
        assert ratio.hi is not None
        assert ratio.lo <= ratio.hi
        assert ratio.lo <= cast(float, ratio.estimate) <= ratio.hi

    def test_missing_hi_raises(self) -> None:
        with pytest.raises(ValueError):
            _ratio(estimate=1.5, lo=1.0, hi=None, shape=IntervalShape.BOUNDED)

    def test_lo_greater_than_hi_raises(self) -> None:
        with pytest.raises(ValueError):
            _ratio(estimate=1.5, lo=2.0, hi=1.0, shape=IntervalShape.BOUNDED)

    def test_estimate_outside_lo_hi_raises(self) -> None:
        with pytest.raises(ValueError):
            _ratio(estimate=5.0, lo=1.0, hi=2.0, shape=IntervalShape.BOUNDED)


class TestRatioShapeUnbounded:
    def test_valid_construction_has_no_lo_or_hi(self) -> None:
        # UNBOUNDED still accepts (in fact requires, as for every OK ratio) an estimate.
        # estimate=1.0 matches _ratio()'s default threshold_a/threshold_b ratio.
        ratio = _ratio(estimate=1.0, lo=None, hi=None, shape=IntervalShape.UNBOUNDED)

        assert ratio.lo is None
        assert ratio.hi is None

    def test_lo_present_raises(self) -> None:
        with pytest.raises(ValueError):
            _ratio(estimate=1.5, lo=1.0, hi=None, shape=IntervalShape.UNBOUNDED)

    def test_hi_present_raises(self) -> None:
        with pytest.raises(ValueError):
            _ratio(estimate=1.5, lo=None, hi=1.0, shape=IntervalShape.UNBOUNDED)


class TestRatioShapeExclusive:
    """``(-inf, lo] union [hi, inf)``: the two rays never meet, so ``lo < hi`` strictly."""

    def test_valid_construction(self) -> None:
        # An OK ratio requires a non-None estimate even for EXCLUSIVE, and an EXCLUSIVE one
        # requires it to lie outside (lo, hi). estimate=1.0 matches _ratio()'s default threshold
        # ratio and sits at the closed lo edge (1.0 <= lo is false here; instead hi=1.0 <=
        # estimate), which is the ">= hi" branch of "estimate <= lo or estimate >= hi".
        ratio = _ratio(estimate=1.0, lo=0.2, hi=1.0, shape=IntervalShape.EXCLUSIVE)

        assert ratio.lo is not None
        assert ratio.hi is not None
        assert ratio.lo < ratio.hi

    def test_lo_equal_to_hi_raises(self) -> None:
        with pytest.raises(ValueError):
            _ratio(estimate=None, lo=1.0, hi=1.0, shape=IntervalShape.EXCLUSIVE)

    def test_missing_lo_raises(self) -> None:
        with pytest.raises(ValueError):
            _ratio(estimate=None, lo=None, hi=2.0, shape=IntervalShape.EXCLUSIVE)


class TestRatioShapeRulesOnlyApplyWhenCensoringIsNone:
    """The shape rules above apply only to a ratio with censoring ``NONE``. A censored or failed
    ratio carries no numbers at all: ``shape`` (like ``estimate``/``lo``/``hi``) must be exactly
    ``None``, not merely unchecked.
    """

    def test_shape_is_none_under_non_none_censoring(self) -> None:
        ratio = _ratio(
            estimate=None,
            lo=None,
            hi=None,
            shape=None,
            censoring=Censoring.RIGHT,
        )

        assert ratio.censoring is Censoring.RIGHT
        assert ratio.shape is None
        assert ratio.lo is None
        assert ratio.hi is None


class TestRatioOkNoneRequiresBothThresholdsOk:
    """For the ``status`` OK + ``censoring`` NONE ratio path, both thresholds must
    themselves have ``status`` OK and a ``value`` set."""

    def test_threshold_a_not_ok_raises(self) -> None:
        not_ok_threshold = _threshold(
            value=None,
            lo=0.02,
            hi=0.09,
            interval_method="exact_bound",
            censoring=Censoring.NONE,
            status=Status.NOT_CONVERGED,
        )

        with pytest.raises(ValueError):
            _ratio(
                estimate=1.0,
                lo=0.5,
                hi=2.0,
                shape=IntervalShape.BOUNDED,
                threshold_a=not_ok_threshold,
            )

    def test_threshold_b_ok_but_without_a_value_raises(self) -> None:
        right_censored_threshold = _threshold(
            value=None,
            lo=10.0,
            hi=None,
            interval_method="exact_bound",
            censoring=Censoring.RIGHT,
            status=Status.OK,
        )

        with pytest.raises(ValueError):
            _ratio(
                estimate=1.0,
                lo=0.5,
                hi=2.0,
                shape=IntervalShape.BOUNDED,
                threshold_b=right_censored_threshold,
            )


class TestRatioOkNoneEstimateRequiredAndMatchesThresholdRatio:
    """``estimate`` is required (not ``None``) and must equal ``threshold_a.value /
    threshold_b.value`` within ``math.isclose(rel_tol=1e-9)``."""

    @staticmethod
    def _matched_thresholds() -> tuple[Threshold, Threshold]:
        # Distinct fits (_CELLS vs _CELLS_B), modelling two disjoint samples.
        t_a = _threshold(
            value=0.1,
            lo=0.05,
            hi=0.15,
            interval_method="profile",
            fit=_fit_with_axis(_AXIS, cells=_CELLS),
        )
        t_b = _threshold(
            value=0.2,
            lo=0.1,
            hi=0.3,
            interval_method="profile",
            fit=_fit_with_axis(_AXIS, cells=_CELLS_B),
        )
        return t_a, t_b

    def test_estimate_none_raises(self) -> None:
        t_a, t_b = self._matched_thresholds()

        with pytest.raises(ValueError):
            _ratio(
                estimate=None,
                lo=0.25,
                hi=0.75,
                shape=IntervalShape.BOUNDED,
                threshold_a=t_a,
                threshold_b=t_b,
            )

    def test_estimate_mismatched_with_threshold_ratio_raises(self) -> None:
        t_a, t_b = self._matched_thresholds()

        with pytest.raises(ValueError):
            _ratio(
                estimate=999.0,
                lo=0.25,
                hi=1500.0,
                shape=IntervalShape.BOUNDED,
                threshold_a=t_a,
                threshold_b=t_b,
            )

    def test_estimate_matching_threshold_ratio_is_accepted(self) -> None:
        t_a, t_b = self._matched_thresholds()

        ratio = _ratio(
            estimate=0.5,
            lo=0.25,
            hi=0.75,
            shape=IntervalShape.BOUNDED,
            threshold_a=t_a,
            threshold_b=t_b,
        )

        assert ratio.estimate == pytest.approx(0.5)


class TestRatioOkNoneExclusiveEstimateOutsideInterval:
    """``EXCLUSIVE`` requires ``estimate <= lo or estimate >= hi``."""

    def test_estimate_inside_the_excluded_gap_raises(self) -> None:
        # Distinct fits, so the ValueError below is unambiguously about the excluded gap, not an
        # incidental shared-fit rejection.
        t_a = _threshold(
            value=1.0,
            lo=0.5,
            hi=1.5,
            interval_method="profile",
            fit=_fit_with_axis(_AXIS, cells=_CELLS),
        )
        t_b = _threshold(
            value=1.0,
            lo=0.5,
            hi=1.5,
            interval_method="profile",
            fit=_fit_with_axis(_AXIS, cells=_CELLS_B),
        )

        with pytest.raises(ValueError):
            _ratio(
                estimate=1.0,
                lo=0.5,
                hi=2.0,
                shape=IntervalShape.EXCLUSIVE,
                threshold_a=t_a,
                threshold_b=t_b,
            )

    def test_estimate_at_or_below_lo_is_accepted(self) -> None:
        # Distinct fits, or this accept test would itself raise on a shared fit.
        t_a = _threshold(
            value=0.5,
            lo=0.3,
            hi=0.7,
            interval_method="profile",
            fit=_fit_with_axis(_AXIS, cells=_CELLS),
        )
        t_b = _threshold(
            value=1.0,
            lo=0.8,
            hi=1.2,
            interval_method="profile",
            fit=_fit_with_axis(_AXIS, cells=_CELLS_B),
        )

        ratio = _ratio(
            estimate=0.5,
            lo=0.5,
            hi=2.0,
            shape=IntervalShape.EXCLUSIVE,
            threshold_a=t_a,
            threshold_b=t_b,
        )

        assert ratio.estimate is not None
        assert ratio.lo is not None
        assert ratio.estimate <= ratio.lo


class TestRatioOkNoneUnboundedAcceptsAnEstimate:
    """``UNBOUNDED`` accepts an estimate (and, as for every OK ratio, requires one)."""

    def test_unbounded_with_an_estimate_is_accepted(self) -> None:
        # Distinct fits, or this accept test would itself raise on a shared fit.
        t_a = _threshold(
            value=1.0,
            lo=0.5,
            hi=1.5,
            interval_method="profile",
            fit=_fit_with_axis(_AXIS, cells=_CELLS),
        )
        t_b = _threshold(
            value=1.0,
            lo=0.5,
            hi=1.5,
            interval_method="profile",
            fit=_fit_with_axis(_AXIS, cells=_CELLS_B),
        )

        ratio = _ratio(
            estimate=1.0,
            lo=None,
            hi=None,
            shape=IntervalShape.UNBOUNDED,
            threshold_a=t_a,
            threshold_b=t_b,
        )

        assert ratio.estimate == pytest.approx(1.0)


class TestRatioOkNoneLogDeltaRequiresBoundedWithPositiveInterval:
    """``method="log_delta"`` requires shape ``BOUNDED`` (with ``0 < lo <= estimate <= hi``)
    **or** ``HALF_OPEN`` (with ``0 < lo <= estimate``, ``hi is None``) -- the same positivity
    requirement extended to the half-open shape, tested at the end of this class. Neither shape
    is generated by ``ratio_interval(method="log_delta")`` itself in this release (it is always
    a finite, exponentiated two-sided interval, so ``BOUNDED`` in practice), but ``Ratio``'s own
    type accepts either, and the positivity requirement -- the reason
    ``log_delta`` needs ``lo > 0`` at all, since a log-scale interval is positive by
    construction -- must hold regardless of which shape carries it."""

    @staticmethod
    def _matched_thresholds() -> tuple[Threshold, Threshold]:
        # Distinct fits (_CELLS vs _CELLS_B).
        t_a = _threshold(
            value=1.0,
            lo=0.5,
            hi=1.5,
            interval_method="profile",
            fit=_fit_with_axis(_AXIS, cells=_CELLS),
        )
        t_b = _threshold(
            value=1.0,
            lo=0.5,
            hi=1.5,
            interval_method="profile",
            fit=_fit_with_axis(_AXIS, cells=_CELLS_B),
        )
        return t_a, t_b

    def test_log_delta_with_unbounded_shape_raises(self) -> None:
        t_a, t_b = self._matched_thresholds()

        with pytest.raises(ValueError):
            _ratio(
                estimate=1.0,
                lo=None,
                hi=None,
                shape=IntervalShape.UNBOUNDED,
                method="log_delta",
                threshold_a=t_a,
                threshold_b=t_b,
            )

    def test_log_delta_with_a_zero_lower_bound_raises(self) -> None:
        t_a, t_b = self._matched_thresholds()

        with pytest.raises(ValueError):
            _ratio(
                estimate=1.0,
                lo=0.0,
                hi=2.0,
                shape=IntervalShape.BOUNDED,
                method="log_delta",
                threshold_a=t_a,
                threshold_b=t_b,
            )

    def test_log_delta_accepts_a_positive_bounded_interval(self) -> None:
        t_a, t_b = self._matched_thresholds()

        ratio = _ratio(
            estimate=1.0,
            lo=0.5,
            hi=2.0,
            shape=IntervalShape.BOUNDED,
            method="log_delta",
            threshold_a=t_a,
            threshold_b=t_b,
        )

        assert ratio.method == "log_delta"

    def test_log_delta_with_half_open_and_a_zero_lower_bound_raises(self) -> None:
        t_a, t_b = self._matched_thresholds()

        with pytest.raises(ValueError):
            _ratio(
                estimate=1.0,
                lo=0.0,
                hi=None,
                shape=IntervalShape.HALF_OPEN,
                method="log_delta",
                threshold_a=t_a,
                threshold_b=t_b,
            )

    def test_log_delta_accepts_a_positive_half_open_interval(self) -> None:
        t_a, t_b = self._matched_thresholds()

        ratio = _ratio(
            estimate=1.0,
            lo=0.5,
            hi=None,
            shape=IntervalShape.HALF_OPEN,
            method="log_delta",
            threshold_a=t_a,
            threshold_b=t_b,
        )

        assert ratio.method == "log_delta"
        assert ratio.shape is IntervalShape.HALF_OPEN
        assert ratio.hi is None


class TestRatioDefinitionsMustMatch:
    """Any ratio (not only the OK/NONE path) requires the two thresholds'
    ``Definition``s to be equal -- a ratio between an ``absolute`` threshold and a
    ``baseline_fraction`` threshold is not a meaningful number."""

    def test_mismatched_definitions_raise(self) -> None:
        # Distinct fits, so the ValueError below is unambiguously about the mismatched
        # definitions, not an incidental shared-fit rejection.
        t_a = _threshold(
            value=0.1,
            lo=0.05,
            hi=0.15,
            interval_method="profile",
            definition=Definition.absolute(0.95),
            fit=_fit_with_axis(_AXIS, cells=_CELLS),
        )
        t_b = _threshold(
            value=0.2,
            lo=0.1,
            hi=0.3,
            interval_method="profile",
            definition=Definition.baseline_fraction(0.5),
            fit=_fit_with_axis(_AXIS, cells=_CELLS_B),
        )

        with pytest.raises(ValueError):
            _ratio(
                estimate=0.5,
                lo=0.25,
                hi=0.75,
                shape=IntervalShape.BOUNDED,
                threshold_a=t_a,
                threshold_b=t_b,
            )

    def test_matching_definitions_is_accepted(self) -> None:
        # Distinct fits, or this accept test would itself raise on a shared fit.
        t_a = _threshold(
            value=0.1,
            lo=0.05,
            hi=0.15,
            interval_method="profile",
            fit=_fit_with_axis(_AXIS, cells=_CELLS),
        )
        t_b = _threshold(
            value=0.2,
            lo=0.1,
            hi=0.3,
            interval_method="profile",
            fit=_fit_with_axis(_AXIS, cells=_CELLS_B),
        )

        ratio = _ratio(
            estimate=0.5,
            lo=0.25,
            hi=0.75,
            shape=IntervalShape.BOUNDED,
            threshold_a=t_a,
            threshold_b=t_b,
        )

        assert ratio.threshold_a.definition == ratio.threshold_b.definition


def _fit_with_cluster_ids(cluster_ids: tuple[str, ...]) -> Fit:
    return Fit(
        axis=_AXIS,
        outcome="success",
        direction="decreasing",
        model="binomial",
        link="probit",
        params=_params(),
        covariance=_covariance(),
        log_likelihood=-1.0,
        cells=_CELLS,
        status=Status.OK,
        cluster_ids=cluster_ids,
    )


class TestRatioIndependentDependenceForbidsOverlappingClusterIds:
    """``dependence="independent"`` is valid only when the
    two thresholds come from disjoint data. ``Fit.cluster_ids`` is what lets that be checked
    directly, rather than only by convention."""

    def test_overlapping_cluster_ids_raises(self) -> None:
        t_a = _threshold(
            value=0.1,
            lo=0.05,
            hi=0.15,
            interval_method="profile",
            fit=_fit_with_cluster_ids(("s1", "s2")),
        )
        t_b = _threshold(
            value=0.2,
            lo=0.1,
            hi=0.3,
            interval_method="profile",
            fit=_fit_with_cluster_ids(("s2", "s3")),
        )

        with pytest.raises(ValueError):
            _ratio(
                estimate=0.5,
                lo=0.25,
                hi=0.75,
                shape=IntervalShape.BOUNDED,
                threshold_a=t_a,
                threshold_b=t_b,
                dependence="independent",
            )

    def test_disjoint_cluster_ids_are_accepted(self) -> None:
        t_a = _threshold(
            value=0.1,
            lo=0.05,
            hi=0.15,
            interval_method="profile",
            fit=_fit_with_cluster_ids(("s1", "s2")),
        )
        t_b = _threshold(
            value=0.2,
            lo=0.1,
            hi=0.3,
            interval_method="profile",
            fit=_fit_with_cluster_ids(("s3", "s4")),
        )

        ratio = _ratio(
            estimate=0.5,
            lo=0.25,
            hi=0.75,
            shape=IntervalShape.BOUNDED,
            threshold_a=t_a,
            threshold_b=t_b,
            dependence="independent",
        )

        assert ratio.dependence == "independent"

    def test_one_side_with_no_cluster_ids_is_accepted(self) -> None:
        t_a = _threshold(value=0.1, lo=0.05, hi=0.15, interval_method="profile")
        t_b = _threshold(
            value=0.2,
            lo=0.1,
            hi=0.3,
            interval_method="profile",
            fit=_fit_with_cluster_ids(("s1", "s2")),
        )

        ratio = _ratio(
            estimate=0.5,
            lo=0.25,
            hi=0.75,
            shape=IntervalShape.BOUNDED,
            threshold_a=t_a,
            threshold_b=t_b,
            dependence="independent",
        )

        assert ratio.dependence == "independent"


class TestRatioMethodDependenceLevel:
    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError):
            _ratio(estimate=1.0, lo=0.5, hi=2.0, shape=IntervalShape.BOUNDED, method="bootstrap")

    @pytest.mark.parametrize("method", ["fieller", "log_delta"])
    def test_documented_methods_are_accepted(self, method: str) -> None:
        ratio = _ratio(estimate=1.0, lo=0.5, hi=2.0, shape=IntervalShape.BOUNDED, method=method)

        assert ratio.method == method

    def test_dependence_other_than_independent_raises(self) -> None:
        """``dependence="paired"`` is not implemented; only ``"independent"`` is allowed."""
        with pytest.raises(ValueError):
            _ratio(
                estimate=1.0,
                lo=0.5,
                hi=2.0,
                shape=IntervalShape.BOUNDED,
                dependence="paired",
            )

    @pytest.mark.parametrize("level", [0.0, 1.0])
    def test_level_at_the_endpoints_raises(self, level: float) -> None:
        with pytest.raises(ValueError):
            _ratio(estimate=1.0, lo=0.5, hi=2.0, shape=IntervalShape.BOUNDED, level=level)


class TestRatioDefaultsAndFrozen:
    def test_defaults(self) -> None:
        ratio = _ratio(estimate=1.0, lo=0.5, hi=2.0, shape=IntervalShape.BOUNDED)

        assert ratio.warnings == ()
        # The package-wide SCHEMA_VERSION, currently "1".
        assert ratio.schema_version == "1"
        assert ratio.provenance == {}

    def test_is_frozen(self) -> None:
        ratio = _ratio(estimate=1.0, lo=0.5, hi=2.0, shape=IntervalShape.BOUNDED)

        with pytest.raises(dataclasses.FrozenInstanceError):
            cast(Any, ratio).estimate = 2.0


# ------------------------------------------------------------------------------------------
# Number fields reject bool and non-finite values at construction.
# ------------------------------------------------------------------------------------------


class TestFitLogLikelihoodRejectsBoolAndNonFinite:
    def test_bool_log_likelihood_raises(self) -> None:
        """``math.isfinite(True)`` is ``True``, so the existing finiteness check alone does not
        reject a bool ``log_likelihood``."""
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params=_params(),
                covariance=_covariance(),
                log_likelihood=True,
                cells=_CELLS,
                status=Status.OK,
            )

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_log_likelihood_raises(self, value: float) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params=_params(),
                covariance=_covariance(),
                log_likelihood=value,
                cells=_CELLS,
                status=Status.OK,
            )


class TestThresholdNumberFieldsRejectBool:
    """``value``/``lo``/``hi`` are chosen so every *other* Threshold invariant (log-axis
    positivity, ``lo <= value <= hi``) is already satisfied once the bool is read as its ``int``
    equivalent -- isolating the bool check specifically, rather than incidentally tripping over
    an ordering or positivity rule instead."""

    def test_bool_value_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(value=True, lo=0.5, hi=2.0, interval_method="profile")

    def test_bool_lo_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(value=1.0, lo=True, hi=2.0, interval_method="profile")

    def test_bool_hi_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(value=1.0, lo=0.5, hi=True, interval_method="profile")

    def test_bool_level_raises(self) -> None:
        with pytest.raises(ValueError):
            _threshold(value=0.05, lo=0.03, hi=0.08, interval_method="profile", level=True)


class TestRatioNumberFieldsRejectBool:
    """As with ``Threshold``, ``estimate``/``lo``/``hi`` line up with the default thresholds'
    ratio of 1.0 so the bool (read as its ``int`` equivalent) would otherwise satisfy every
    other BOUNDED-shape rule."""

    def test_bool_estimate_raises(self) -> None:
        with pytest.raises(ValueError):
            _ratio(estimate=True, lo=0.5, hi=2.0, shape=IntervalShape.BOUNDED)

    def test_bool_lo_raises(self) -> None:
        with pytest.raises(ValueError):
            _ratio(estimate=1.0, lo=True, hi=2.0, shape=IntervalShape.BOUNDED)

    def test_bool_hi_raises(self) -> None:
        with pytest.raises(ValueError):
            _ratio(estimate=1.0, lo=0.5, hi=True, shape=IntervalShape.BOUNDED)

    def test_bool_level_raises(self) -> None:
        with pytest.raises(ValueError):
            _ratio(estimate=1.0, lo=0.5, hi=2.0, shape=IntervalShape.BOUNDED, level=True)


class TestCovarianceMatrixRejectsBoolEntry:
    def test_bool_entry_raises(self) -> None:
        """A 1x1 matrix isolates the bool check: ``True`` read as ``1.0`` would otherwise be a
        perfectly ordinary, positive-definite single-entry covariance."""
        with pytest.raises(ValueError):
            Covariance(names=("mu",), matrix=((True,),))


# ------------------------------------------------------------------------------------------
# Ratio invariants.
# ------------------------------------------------------------------------------------------


class TestRatioIndependentDependenceAcceptsEqualFits:
    """With ``dependence="independent"``, value-equal fits are **accepted**.

    Two disjoint samples can produce identical counts and so identical fits, and a ratio between
    them is valid. An equality check on the two fits would reject that valid case, and could be
    bypassed by editing ``provenance``. The object-level guard is overlapping ``cluster_ids``
    only; the full shared-data check belongs to ``ratio_interval``."""

    def test_value_equal_fits_from_disjoint_samples_are_accepted(self) -> None:
        fit_a = _fit_with_axis(_AXIS)
        fit_b = _fit_with_axis(_AXIS)
        assert fit_a == fit_b  # sanity: same axis/status/direction/cells, so value-equal

        t_a = _threshold(value=0.1, lo=0.05, hi=0.15, interval_method="profile", fit=fit_a)
        t_b = _threshold(value=0.2, lo=0.1, hi=0.3, interval_method="profile", fit=fit_b)

        ratio = _ratio(
            estimate=0.5,
            lo=0.25,
            hi=0.75,
            shape=IntervalShape.BOUNDED,
            threshold_a=t_a,
            threshold_b=t_b,
            dependence="independent",
        )

        assert ratio.estimate == 0.5

    def test_distinct_fits_are_accepted(self) -> None:
        t_a = _threshold(
            value=0.1,
            lo=0.05,
            hi=0.15,
            interval_method="profile",
            fit=_fit_with_axis(_AXIS, cells=_CELLS),
        )
        t_b = _threshold(
            value=0.2,
            lo=0.1,
            hi=0.3,
            interval_method="profile",
            fit=_fit_with_axis(_AXIS, cells=_CELLS_B),
        )

        ratio = _ratio(
            estimate=0.5,
            lo=0.25,
            hi=0.75,
            shape=IntervalShape.BOUNDED,
            threshold_a=t_a,
            threshold_b=t_b,
            dependence="independent",
        )

        assert ratio.threshold_a.fit != ratio.threshold_b.fit


class TestRatioOkNoneBothThresholdsMustHaveNoneCensoring:
    """On the ratio's own status OK + censoring NONE path, both input thresholds must
    themselves have censoring NONE -- an OPEN_UPPER or OPEN_LOWER input threshold raises,
    even though it independently satisfies the "status OK with value set" check."""

    def test_threshold_a_open_upper_raises(self) -> None:
        t_a = _threshold(
            value=0.05,
            lo=0.03,
            hi=None,
            interval_method="profile",
            censoring=Censoring.OPEN_UPPER,
            status=Status.OK,
        )

        with pytest.raises(ValueError):
            _ratio(
                estimate=1.0,
                lo=0.5,
                hi=2.0,
                shape=IntervalShape.BOUNDED,
                threshold_a=t_a,
                status=Status.OK,
                censoring=Censoring.NONE,
            )

    def test_threshold_b_open_lower_raises(self) -> None:
        t_b = _threshold(
            value=0.05,
            lo=None,
            hi=0.08,
            interval_method="profile",
            censoring=Censoring.OPEN_LOWER,
            status=Status.OK,
        )

        with pytest.raises(ValueError):
            _ratio(
                estimate=1.0,
                lo=0.5,
                hi=2.0,
                shape=IntervalShape.BOUNDED,
                threshold_b=t_b,
                status=Status.OK,
                censoring=Censoring.NONE,
            )


class TestRatioOkNoneZeroDenominatorRaisesValueErrorNotZeroDivisionError:
    """``threshold_b.value > 0`` is checked before ever dividing, so a ``0.0``
    denominator on a linear axis raises ``ValueError`` -- never Python's own
    ``ZeroDivisionError``."""

    def test_zero_denominator_on_linear_axis_raises_value_error_not_zero_division_error(
        self,
    ) -> None:
        linear_axis = Axis(name="axis_2", unit="m/s", scale="linear")
        t_a = _threshold(
            value=0.5,
            lo=0.3,
            hi=0.7,
            interval_method="profile",
            axis=linear_axis,
            fit=_fit_with_axis(linear_axis, cells=_CELLS),
        )
        t_b = _threshold(
            value=0.0,
            lo=0.0,
            hi=0.1,
            interval_method="profile",
            axis=linear_axis,
            fit=_fit_with_axis(linear_axis, cells=_CELLS_B),
        )

        try:
            _ratio(
                estimate=1.0,
                lo=0.5,
                hi=2.0,
                shape=IntervalShape.BOUNDED,
                threshold_a=t_a,
                threshold_b=t_b,
            )
        except ZeroDivisionError:
            pytest.fail("threshold_b.value == 0.0 must raise ValueError, not ZeroDivisionError")
        except ValueError:
            pass
        else:
            pytest.fail("expected a ValueError for a zero denominator")


class TestRatioLogDeltaRequiresALogAxis:
    """``method="log_delta"`` requires both thresholds on a log axis, in addition
    to the BOUNDED-shape and ``0 < lo <= estimate <= hi`` rules -- a linear axis
    with an otherwise perfectly valid positive bounded interval still raises."""

    def test_log_delta_on_a_linear_axis_raises(self) -> None:
        linear_axis = Axis(name="axis_2", unit="m/s", scale="linear")
        t_a = _threshold(
            value=1.0,
            lo=0.5,
            hi=1.5,
            interval_method="profile",
            axis=linear_axis,
            fit=_fit_with_axis(linear_axis, cells=_CELLS),
        )
        t_b = _threshold(
            value=1.0,
            lo=0.5,
            hi=1.5,
            interval_method="profile",
            axis=linear_axis,
            fit=_fit_with_axis(linear_axis, cells=_CELLS_B),
        )

        with pytest.raises(ValueError):
            _ratio(
                estimate=1.0,
                lo=0.5,
                hi=2.0,
                shape=IntervalShape.BOUNDED,
                method="log_delta",
                threshold_a=t_a,
                threshold_b=t_b,
            )

    def test_log_delta_on_a_log_axis_is_accepted(self) -> None:
        t_a = _threshold(
            value=1.0,
            lo=0.5,
            hi=1.5,
            interval_method="profile",
            fit=_fit_with_axis(_AXIS, cells=_CELLS),
        )
        t_b = _threshold(
            value=1.0,
            lo=0.5,
            hi=1.5,
            interval_method="profile",
            fit=_fit_with_axis(_AXIS, cells=_CELLS_B),
        )

        ratio = _ratio(
            estimate=1.0,
            lo=0.5,
            hi=2.0,
            shape=IntervalShape.BOUNDED,
            method="log_delta",
            threshold_a=t_a,
            threshold_b=t_b,
        )

        assert ratio.method == "log_delta"


# ------------------------------------------------------------------------------------------
# Threshold.
# ------------------------------------------------------------------------------------------


class TestThresholdTerminalStatusRequiresExactBoundIntervalMethod:
    """A threshold with status UNREACHABLE, FAILS_AT_BASELINE or CONTROL_INCOMPATIBLE requires
    ``interval_method == "exact_bound"``, even though ``"profile"`` is itself a recognised
    ``interval_method`` value (so the pre-existing "unknown interval_method" check alone does
    not catch this)."""

    @pytest.mark.parametrize(
        "status", [Status.UNREACHABLE, Status.FAILS_AT_BASELINE, Status.CONTROL_INCOMPATIBLE]
    )
    def test_non_exact_bound_interval_method_raises(self, status: Status) -> None:
        with pytest.raises(ValueError):
            _threshold(
                value=None,
                lo=None,
                hi=None,
                interval_method="profile",
                censoring=Censoring.NONE,
                status=status,
            )


# ------------------------------------------------------------------------------------------
# Fit and Covariance.
# ------------------------------------------------------------------------------------------


class TestCovarianceMustBePositiveDefiniteNotJustSemiDefinite:
    def test_singular_matrix_with_a_zero_eigenvalue_raises(self) -> None:
        with pytest.raises(ValueError):
            Covariance(names=("mu", "s"), matrix=((0.0, 0.0), (0.0, 1.0)))

    def test_rank_deficient_matrix_raises(self) -> None:
        with pytest.raises(ValueError):
            Covariance(names=("mu", "s"), matrix=((1.0, 1.0), (1.0, 1.0)))

    def test_error_message_says_positive_definite_not_semi_definite(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            Covariance(names=("mu", "s"), matrix=((0.0, 0.0), (0.0, 1.0)))

        message = str(exc_info.value)
        assert "positive-definite" in message or "positive definite" in message
        assert "semi-definite" not in message
        assert "semi definite" not in message


class TestFitRequiresMuAndSNotFixedWhenOk:
    def test_fixed_mu_raises(self) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params={
                    "mu": Parameter(value=0.0, fixed=True),
                    "s": Parameter(value=1.0, fixed=False),
                    "upper": Parameter(value=1.0, fixed=True),
                    "lower": Parameter(value=0.0, fixed=True),
                },
                covariance=Covariance(names=("s",), matrix=((1.0,),)),
                log_likelihood=-1.0,
                cells=_CELLS,
                status=Status.OK,
            )

    def test_fixed_s_raises(self) -> None:
        with pytest.raises(ValueError):
            Fit(
                axis=_AXIS,
                outcome="success",
                direction="decreasing",
                model="binomial",
                link="probit",
                params={
                    "mu": Parameter(value=0.0, fixed=False),
                    "s": Parameter(value=1.0, fixed=True),
                    "upper": Parameter(value=1.0, fixed=True),
                    "lower": Parameter(value=0.0, fixed=True),
                },
                covariance=Covariance(names=("mu",), matrix=((1.0,),)),
                log_likelihood=-1.0,
                cells=_CELLS,
                status=Status.OK,
            )

    def test_mu_and_s_not_fixed_is_accepted(self) -> None:
        fit = _ok_fit()

        assert fit.params is not None
        assert fit.params["mu"].fixed is False
        assert fit.params["s"].fixed is False


class TestFitLinkTypeAnnotationIsStrNotOptional:
    """``Fit.link``'s schema enum has no ``null`` (exercised against the packaged
    schema in ``tests/unit/test_report.py``), and its Python type annotation must match: ``str``,
    not ``str | None`` -- ``Fit.__post_init__`` already rejects ``link=None`` at runtime (it is
    not one of the three recognised link names), so the optional annotation was never accurate.
    """

    def test_link_field_type_annotation_is_str(self) -> None:
        link_field = next(f for f in dataclasses.fields(Fit) if f.name == "link")

        assert link_field.type == "str"
