"""``ratio_interval()``'s guards: the checks that must raise before -- or
instead of -- any Fieller/log-delta computation happens.

Four of these mirror invariants ``Ratio.__post_init__`` already enforces (axis name, axis unit,
``Definition`` equality, overlapping cluster ids under ``dependence="independent"`` --
``tests/unit/test_results.py``'s ``TestRatioAxisGuard``, ``TestRatioDefinitionsMustMatch``,
``TestRatioIndependentDependenceForbidsOverlappingClusterIds``), so this file is not proving
those rules exist -- it is proving ``ratio_interval()``, the public entry point real callers use,
actually surfaces them (whether by constructing a ``Ratio`` and letting its own validation fire,
or by checking earlier) rather than silently swallowing the mismatch or crashing with something
other than ``ValueError``.

Two guards have no ``Ratio``-level equivalent, because they are about the *call*, not the
resulting object: ``dependence="paired"`` (not implemented -- mirrors ``threshold()``'s own
guard, ``threshold.py``'s ``dependence == "paired"`` check) and ``level`` inheritance
(``level=None`` takes the level from the two input thresholds, which must agree; an explicit
``level`` that contradicts them also raises -- no silent fallback to a value the caller did not
ask for).

This file constructs ``Fit``/``Threshold`` objects directly rather than fitting real data,
mirroring ``tests/unit/test_results.py``'s own helpers (``_params``, ``_covariance``,
``_fit_with_axis``, ``_threshold``): the guards under test do not depend on what a real fit
looks like, only on the axis/definition/cluster-id/level relationship between two otherwise
unremarkable ``OK`` thresholds.
"""

from __future__ import annotations

import math

import pytest

from robustmargin import (
    Axis,
    Cell,
    Censoring,
    Definition,
    Fit,
    Status,
    Threshold,
    ratio_interval,
)
from robustmargin.models import Covariance, Parameter

_AXIS = Axis(name="noise", unit="m", scale="log")
_CELLS = (
    Cell(severity=0.0, successes=100, trials=100),
    Cell(severity=1.0, successes=40, trials=100),
)


def _params(*, mu: float = 0.0, s: float = 1.0) -> dict[str, Parameter]:
    return {
        "mu": Parameter(value=mu, fixed=False),
        "s": Parameter(value=s, fixed=False),
        "upper": Parameter(value=1.0, fixed=True),
        "lower": Parameter(value=0.0, fixed=True),
    }


def _covariance() -> Covariance:
    return Covariance(names=("mu", "s"), matrix=((0.01, 0.0), (0.0, 0.02)))


def _fit(
    *,
    axis: Axis = _AXIS,
    mu: float = 0.0,
    cluster_ids: tuple[str, ...] | None = None,
) -> Fit:
    return Fit(
        axis=axis,
        outcome="success",
        direction="decreasing",
        model="binomial",
        link="probit",
        params=_params(mu=mu),
        covariance=_covariance(),
        log_likelihood=-42.0,
        cells=_CELLS,
        status=Status.OK,
        cluster_ids=cluster_ids,
    )


def _threshold(
    *,
    axis: Axis = _AXIS,
    definition: Definition | None = None,
    value: float = 1.0,
    level: float = 0.95,
    cluster_ids: tuple[str, ...] | None = None,
    fit: Fit | None = None,
) -> Threshold:
    if fit is None:
        fit = _fit(
            axis=axis,
            mu=math.log(value) if axis.scale == "log" else value,
            cluster_ids=cluster_ids,
        )
    return Threshold(
        axis=axis,
        definition=definition if definition is not None else Definition.absolute(0.5),
        direction="decreasing",
        value=value,
        lo=value * 0.5,
        hi=value * 2.0,
        level=level,
        interval_method="delta",
        censoring=Censoring.NONE,
        status=Status.OK,
        dependence="independent",
        fit=fit,
    )


class TestAxisNameGuard:
    def test_mismatched_axis_name_raises(self) -> None:
        t_a = _threshold(axis=_AXIS)
        t_b = _threshold(axis=Axis(name="axis_2", unit="m", scale="log"))

        with pytest.raises(ValueError):
            ratio_interval(t_a, t_b, method="fieller", dependence="independent")


class TestAxisUnitGuard:
    def test_mismatched_axis_unit_raises(self) -> None:
        t_a = _threshold(axis=_AXIS)
        t_b = _threshold(axis=Axis(name="noise", unit="dB", scale="log"))

        with pytest.raises(ValueError):
            ratio_interval(t_a, t_b, method="fieller", dependence="independent")


class TestDefinitionGuard:
    def test_mismatched_definition_raises(self) -> None:
        t_a = _threshold(definition=Definition.absolute(0.5))
        t_b = _threshold(definition=Definition.absolute(0.9))

        with pytest.raises(ValueError):
            ratio_interval(t_a, t_b, method="fieller", dependence="independent")


class TestClusterIdGuard:
    def test_shared_cluster_ids_under_independent_dependence_raises(self) -> None:
        t_a = _threshold(cluster_ids=("s1", "s2"))
        t_b = _threshold(cluster_ids=("s2", "s3"))

        with pytest.raises(ValueError):
            ratio_interval(t_a, t_b, method="fieller", dependence="independent")

    def test_disjoint_cluster_ids_under_independent_dependence_is_accepted(self) -> None:
        t_a = _threshold(cluster_ids=("s1", "s2"))
        t_b = _threshold(cluster_ids=("s3", "s4"))

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        assert result.dependence == "independent"


class TestPairedDependenceNotImplemented:
    """``dependence="paired"`` is not implemented in this release.

    ``threshold()`` raises ``ValueError`` for the same request (``threshold.py``); this mirrors
    that choice for ``ratio_interval()`` rather than picking ``NotImplementedError``, so the two
    functions are consistent about how an unimplemented option is reported.
    """

    def test_dependence_paired_raises_value_error(self) -> None:
        t_a = _threshold()
        t_b = _threshold(value=2.0)

        with pytest.raises(ValueError):
            ratio_interval(t_a, t_b, method="fieller", dependence="paired")


class TestLevelInheritance:
    """Not a ``Ratio``-level rule: ``level=None`` takes the level from the two input
    thresholds, which must agree; passing an explicit ``level`` that contradicts them also
    raises. Silently defaulting to one side's level, or to some other value the caller never
    asked for, would be a silent fallback."""

    def test_level_none_is_inherited_when_both_thresholds_agree(self) -> None:
        t_a = _threshold(level=0.95)
        t_b = _threshold(value=2.0, level=0.95)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        assert result.level == 0.95

    def test_level_none_with_disagreeing_thresholds_raises(self) -> None:
        t_a = _threshold(level=0.95)
        t_b = _threshold(value=2.0, level=0.90)

        with pytest.raises(ValueError):
            ratio_interval(t_a, t_b, method="fieller", dependence="independent")

    def test_explicit_level_contradicting_both_thresholds_raises(self) -> None:
        t_a = _threshold(level=0.95)
        t_b = _threshold(value=2.0, level=0.95)

        with pytest.raises(ValueError):
            ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=0.90)

    def test_explicit_level_matching_both_thresholds_is_accepted(self) -> None:
        t_a = _threshold(level=0.95)
        t_b = _threshold(value=2.0, level=0.95)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent", level=0.95)

        assert result.level == 0.95


class TestSameFitObjectGuard:
    """The cluster-id check (``TestClusterIdGuard`` above) cannot see two thresholds drawn off the
    *same* ``Fit`` object when that fit carries no cluster ids at all -- ``fit.cluster_ids is None``
    on both sides, so the overlap check trivially passes, yet the ratio would be identically ``1``
    with zero true variance (same number, same data). ``t_a.fit is t_b.fit`` needs no new public
    field and cannot be produced by two genuinely disjoint samples, so it is checked directly.

    This does **not** conflict with ``Ratio``'s own docstring ("equal fits are not rejected: two
    disjoint samples can produce identical counts") -- that sentence is about *value* equality,
    and object identity is strictly stronger. ``test_value_equal_but_distinct_fits_...`` below is
    the case that sentence protects, and it must keep working.
    """

    def test_sharing_the_same_fit_object_under_independent_dependence_raises(self) -> None:
        shared_fit = _fit(mu=0.0)
        t_a = _threshold(value=1.0, fit=shared_fit)
        t_b = _threshold(value=1.0, fit=shared_fit)
        assert t_a.fit is t_b.fit

        with pytest.raises(ValueError):
            ratio_interval(t_a, t_b, method="fieller", dependence="independent")

    def test_value_equal_but_distinct_fits_from_disjoint_samples_are_accepted(self) -> None:
        """Two separately constructed ``Fit`` objects that happen to carry identical values
        (``fit_a == fit_b`` by the dataclass's own field-by-field equality) but are not the same
        object (``fit_a is not fit_b``) model two disjoint samples that produced the same counts
        -- a value-equality check would reject that valid case, so there is none (``Ratio``'s
        own docstring, quoted above)."""
        fit_a = _fit(mu=0.0)
        fit_b = _fit(mu=0.0)
        assert fit_a == fit_b
        assert fit_a is not fit_b
        t_a = _threshold(value=1.0, fit=fit_a)
        t_b = _threshold(value=1.0, fit=fit_b)

        result = ratio_interval(t_a, t_b, method="fieller", dependence="independent")

        assert result.dependence == "independent"


def _threshold_with_log_scale_se(
    *, value: float, sigma_log: float, level: float = 0.95
) -> Threshold:
    """A ``status=OK``, ``censoring=NONE`` threshold with a known log-scale delta standard
    error, for constructing a combined ``log_delta`` variance large enough to overflow
    (the ``log_delta`` overflow guard). Mirrors ``tests/unit/test_ratio_interval_shape.py``'s
    ``_fit_with_known_natural_variance`` / ``_threshold_with_known_natural_variance``: ``z*=0``
    at ``absolute(0.5)`` with fixed ``upper=1, lower=0`` on a symmetric link, so
    ``Cov(mu, mu) = sigma_log**2`` reproduces ``Var(log(value)) = sigma_log**2`` exactly.
    """
    fit = Fit(
        axis=_AXIS,
        outcome="success",
        direction="decreasing",
        model="binomial",
        link="logit",
        params={
            "mu": Parameter(value=math.log(value), fixed=False),
            "s": Parameter(value=1.0, fixed=False),
            "upper": Parameter(value=1.0, fixed=True),
            "lower": Parameter(value=0.0, fixed=True),
        },
        covariance=Covariance(names=("mu", "s"), matrix=((sigma_log**2, 0.0), (0.0, 1.0))),
        log_likelihood=-10.0,
        cells=_CELLS,
        status=Status.OK,
    )
    return Threshold(
        axis=_AXIS,
        definition=Definition.absolute(0.5),
        direction="decreasing",
        value=value,
        lo=value * math.exp(-sigma_log),
        hi=value * math.exp(sigma_log),
        level=level,
        interval_method="delta",
        censoring=Censoring.NONE,
        status=Status.OK,
        dependence="independent",
        fit=fit,
    )


class TestLogDeltaOverflowRaises:
    """When the combined log-scale standard error is too large to
    exponentiate (``sigma_log(ratio) > ~357`` at ``level=0.95``, the point past which
    ``z * sigma_log(ratio)`` exceeds ``intervals.py``'s own ``_MAX_LOG_HALF_WIDTH = 700``),
    ``method="log_delta"`` raises ``ValueError`` naming the standard error and pointing at
    ``method="fieller"`` -- there is no legal non-``BOUNDED`` shape ``Ratio`` can report for this
    method, so a flag is not an option here (unlike the Fieller shapes, where one is).
    """

    def test_log_delta_raises_on_combined_standard_error_overflow(self) -> None:
        # sigma_log_a = sigma_log_b = 260 gives a combined sigma_log(ratio) = sqrt(260^2*2) =
        # ~367.7, comfortably past the ~357.1 threshold (z * 367.7 ~= 720.7 > 700).
        t_a = _threshold_with_log_scale_se(value=1.0, sigma_log=260.0)
        t_b = _threshold_with_log_scale_se(value=2.0, sigma_log=260.0)

        with pytest.raises(ValueError, match="(?i)fieller"):
            ratio_interval(t_a, t_b, method="log_delta", dependence="independent", level=0.95)


class TestLogDeltaUnderflowRaisesFromRatioIntervalNotFromRatio:
    """Regression: a ``log_delta`` half-width can sit *under* ``_MAX_LOG_HALF_WIDTH`` (700, the
    overflow guard ``TestLogDeltaOverflowRaises`` above exercises) and still underflow the
    exponentiated ``lo`` to exactly ``0.0`` -- ``math.isfinite(0.0)`` is ``True``, so
    ``ratio_interval``'s own check would pass it and the failure would only surface downstream, from
    ``Ratio.__post_init__``'s generic ``method='log_delta' requires lo > 0`` invariant, with none of
    the "there is no defensible bounded interval, use method='fieller'" context. ``ratio_interval``
    must be the one to raise, not ``Ratio``.

    **Reachable through the public API**, verified directly (not a below-the-public-API
    construction): splitting the variance across *both* inputs, rather than putting it all on
    one side, keeps each individual ``Threshold`` constructible (a single side carrying all of
    it would underflow *that* side's own ``lo``, which ``Threshold`` itself already rejects on a
    log axis). With ``sigma_log = 357 / sqrt(2) ~= 252.4`` on each side and
    ``value_a=1e-20, value_b=1.0``: each input's own delta-method ``lo`` (``~1.3e-235`` and
    ``~1.3e-215`` respectively) is a normal, representable float, but the combined
    ``sigma_log_ratio = sqrt(252.4^2 + 252.4^2) = 357`` gives ``half_width = z*357 ~= 699.7``
    (under the 700 guard, so the overflow check does not fire) while
    ``lo = 1e-20 * exp(-699.7)`` underflows to exactly ``0.0`` -- confirmed empirically before
    writing this assertion, not assumed from the arithmetic alone.
    """

    def test_underflowed_lo_raises_from_ratio_interval_with_the_endpoint_reason(self) -> None:
        sigma_log = 357.0 / math.sqrt(2.0)
        t_a = _threshold_with_log_scale_se(value=1e-20, sigma_log=sigma_log)
        t_b = _threshold_with_log_scale_se(value=1.0, sigma_log=sigma_log)

        with pytest.raises(ValueError) as exc_info:
            ratio_interval(t_a, t_b, method="log_delta", dependence="independent", level=0.95)

        message = str(exc_info.value)
        # Substance, not exact wording: the message must be about
        # ratio_interval's own endpoint/interval reasoning, not Ratio's generic invariant name.
        assert "ratio_interval" in message
        assert "Ratio.lo must be > 0" not in message
        assert len(message.strip()) > 20
