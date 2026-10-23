"""``Definition.baseline_multiple(k)``: construction, and the mirrored target and gradient the
estimated-asymptote paths need.

``baseline_multiple(k)`` is the severity where an increasing outcome rises to ``k`` times its rate
at zero severity, ``P* = k * P(0) = k * lower``. It needs ``k > 1`` and is refused with a fit of
the other direction (the pairing refusals are in ``test_threshold_increasing.py``).

Mirrored onto the decreasing view (``u' = 1 - l``, ``l' = 1 - u``), the target is
``P'* = 1 - k * l = 1 - k * (1 - u')``, which depends on ``u'``. The two private helpers that
carry a target through the estimated-asymptote profile and the delta gradient therefore need a
branch for it. They are called here with the **mirrored** asymptotes, as the decreasing view does,
and the gradient is held to central finite differences of
``q = (u' - P'*) / (u' - l')``.
"""

from __future__ import annotations

import math

import pytest

from robustmargin import Definition
from robustmargin.intervals import _dq_du_dl, _target_for


class TestConstruction:
    @pytest.mark.parametrize("k", [1.0001, 1.5, 2, 2.0, 10.0, 1e6])
    def test_a_finite_multiple_above_one_is_accepted_and_stored(self, k: float) -> None:
        definition = Definition.baseline_multiple(k)

        assert definition.kind == "baseline_multiple"
        assert definition.value == k

    @pytest.mark.parametrize("k", [1.0, 0.999, 0.5, 0.0, -2.0])
    def test_a_multiple_at_or_below_one_raises_value_error(self, k: float) -> None:
        with pytest.raises(ValueError):
            Definition.baseline_multiple(k)

    @pytest.mark.parametrize("k", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_multiple_raises_value_error(self, k: float) -> None:
        with pytest.raises(ValueError):
            Definition.baseline_multiple(k)

    @pytest.mark.parametrize("k", [True, False])
    def test_a_bool_multiple_raises_value_error(self, k: bool) -> None:
        with pytest.raises(ValueError):
            Definition.baseline_multiple(k)

    def test_the_direct_constructor_enforces_the_same_rule(self) -> None:
        with pytest.raises(ValueError):
            Definition(kind="baseline_multiple", value=1.0)

        assert Definition(kind="baseline_multiple", value=2.0) == Definition.baseline_multiple(2.0)

    def test_it_is_not_equal_to_a_definition_of_another_kind_with_the_same_value(self) -> None:
        assert Definition.baseline_multiple(2.0) != Definition.absolute(2.0)

    def test_the_other_kinds_are_unchanged(self) -> None:
        assert Definition.absolute(0.95).kind == "absolute"
        assert Definition.baseline_fraction(0.5).kind == "baseline_fraction"
        assert Definition.relative(0.5).kind == "relative"
        with pytest.raises(ValueError):
            Definition.baseline_fraction(1.5)


class TestMirroredTarget:
    @pytest.mark.parametrize(
        ("k", "upper", "lower"), [(2.0, 0.9, 0.1), (3.0, 0.95, 0.02), (1.5, 1.0, 0.0)]
    )
    def test_the_target_is_one_minus_k_times_the_original_scale_lower(
        self, k: float, upper: float, lower: float
    ) -> None:
        """``upper``/``lower`` are the mirrored asymptotes ``u', l'``; the original-scale baseline
        is ``l = 1 - u'``."""
        target = _target_for(Definition.baseline_multiple(k), upper=upper, lower=lower)

        assert target == pytest.approx(1.0 - k * (1.0 - upper), rel=1e-12)

    def test_it_depends_on_the_mirrored_upper_asymptote_only(self) -> None:
        definition = Definition.baseline_multiple(2.0)

        assert _target_for(definition, upper=0.9, lower=0.1) == _target_for(
            definition, upper=0.9, lower=0.3
        )
        assert _target_for(definition, upper=0.9, lower=0.1) != _target_for(
            definition, upper=0.8, lower=0.1
        )

    def test_the_other_kinds_are_unchanged(self) -> None:
        assert _target_for(Definition.absolute(0.4), upper=0.9, lower=0.1) == 0.4
        assert _target_for(Definition.baseline_fraction(0.5), upper=0.9, lower=0.1) == 0.45
        assert _target_for(Definition.relative(0.5), upper=0.9, lower=0.1) == pytest.approx(0.5)


def _q(definition: Definition, upper: float, lower: float) -> float:
    return (upper - _target_for(definition, upper=upper, lower=lower)) / (upper - lower)


class TestMirroredGradient:
    @pytest.mark.parametrize("k", [1.2, 2.0, 5.0])
    @pytest.mark.parametrize(("upper", "lower"), [(0.9, 0.1), (0.97, 0.0), (0.99, 0.4)])
    def test_dq_du_dl_matches_central_finite_differences(
        self, k: float, upper: float, lower: float
    ) -> None:
        definition = Definition.baseline_multiple(k)
        h = 1e-6

        dq_du, dq_dl = _dq_du_dl(definition, upper=upper, lower=lower)

        fd_u = (_q(definition, upper + h, lower) - _q(definition, upper - h, lower)) / (2 * h)
        fd_l = (_q(definition, upper, lower + h) - _q(definition, upper, lower - h)) / (2 * h)
        assert dq_du == pytest.approx(fd_u, rel=1e-6, abs=1e-8)
        assert dq_dl == pytest.approx(fd_l, rel=1e-6, abs=1e-8)

    def test_the_branch_is_not_the_relative_zero_gradient(self) -> None:
        """``relative`` has ``q = p`` and so a zero gradient. ``baseline_multiple`` does not: ``q``
        moves with the mirrored upper asymptote."""
        dq_du, _dq_dl = _dq_du_dl(Definition.baseline_multiple(2.0), upper=0.9, lower=0.1)

        assert dq_du != 0.0
