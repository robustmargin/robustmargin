"""``robustmargin.censoring``: the exact one-sided bound arithmetic and per-cell exact rates.

``exact_one_sided_bound`` is the module's one piece of numeric machinery: everything in
``censoring.py``'s classification rules is built from it, so pinning its arithmetic against the
closed forms is what keeps every downstream RIGHT/LEFT/bracket test meaningful.

Each name from ``robustmargin.censoring`` and ``ExactRates``/``per_cell_clopper_pearson``
(``robustmargin.empirical``) is imported **inside** the test that needs it, not at module level,
matching the convention ``tests/contract/`` uses: a module-level
``ModuleNotFoundError``/``ImportError`` would abort collection of this *entire file* (and, by
default, the whole ``pytest`` run), whereas a local import only fails the one test that uses it.
"""

from __future__ import annotations

import math

import pytest
from scipy.stats import binomtest

from robustmargin import (
    Axis,
    Cell,
    Censoring,
    Definition,
    Observations,
    Status,
    Threshold,
    fit_dose_response,
    threshold,
)
from robustmargin.censoring import exact_one_sided_bound
from robustmargin.report import to_dict
from robustmargin.testing import fake_fit


class TestExactOneSidedBound:
    """The closed forms: ``exact_one_sided_bound(k, n, level=c, side=...)`` is
    ``binomtest(k, n).proportion_ci(confidence_level=2*c-1, method="exact")``, read on the
    requested side.
    """

    def test_500_of_500_lower_bound_at_95_matches_the_closed_form(self) -> None:
        from robustmargin.censoring import exact_one_sided_bound

        expected = 0.05 ** (1.0 / 500.0)
        result = exact_one_sided_bound(500, 500, level=0.95, side="lower")
        assert result == pytest.approx(expected, rel=1e-12)
        assert result == pytest.approx(0.9940264484836504, rel=1e-12)

    def test_0_of_300_upper_bound_at_95_matches_the_closed_form(self) -> None:
        from robustmargin.censoring import exact_one_sided_bound

        expected = 1.0 - 0.05 ** (1.0 / 300.0)
        result = exact_one_sided_bound(0, 300, level=0.95, side="upper")
        assert result == pytest.approx(expected, rel=1e-12)
        assert result == pytest.approx(0.00993608194445772, rel=1e-12)

    def test_passing_confidence_level_equal_to_level_does_not_match(self) -> None:
        """This is the 'obvious wrong version': passing
        ``confidence_level=level`` (rather than ``2*level-1``) and reading one side silently
        yields a 97.5% bound where 95% was asked for. Pinned so a regression to it is caught,
        not merely a correct-by-luck coincidence.
        """
        from robustmargin.censoring import exact_one_sided_bound

        wrong = binomtest(500, 500).proportion_ci(confidence_level=0.95, method="exact").low
        correct = exact_one_sided_bound(500, 500, level=0.95, side="lower")

        assert not math.isclose(correct, wrong, rel_tol=1e-9)
        assert correct == pytest.approx(0.05 ** (1.0 / 500.0), rel=1e-12)

    def test_upper_bound_is_at_least_the_observed_rate(self) -> None:
        from robustmargin.censoring import exact_one_sided_bound

        result = exact_one_sided_bound(150, 300, level=0.95, side="upper")
        assert result >= 0.5

    def test_lower_bound_is_at_most_the_observed_rate(self) -> None:
        from robustmargin.censoring import exact_one_sided_bound

        result = exact_one_sided_bound(150, 300, level=0.95, side="lower")
        assert result <= 0.5

    def test_a_wider_level_gives_a_wider_bound(self) -> None:
        """Interval containment/monotonicity: a higher confidence level's
        one-sided bound must be at least as conservative (further from the observed rate).
        """
        from robustmargin.censoring import exact_one_sided_bound

        narrow = exact_one_sided_bound(150, 300, level=0.80, side="upper")
        wide = exact_one_sided_bound(150, 300, level=0.99, side="upper")
        assert wide >= narrow

    def test_side_must_be_lower_or_upper(self) -> None:
        from robustmargin.censoring import exact_one_sided_bound

        with pytest.raises(ValueError):
            exact_one_sided_bound(150, 300, level=0.95, side="both")  # type: ignore[arg-type]


class TestPerCellClopperPearson:
    """``per_cell_clopper_pearson(cells, level=...) -> ExactRates``: takes
    ``Fit.cells`` directly (a ``Sequence[Cell]``), not a ``Fit`` -- ``Fit.cells`` already carries
    severity/successes/trials, so a per-cell result type would only duplicate them.
    """

    def test_returns_parallel_arrays_one_per_cell_with_the_documented_method(self) -> None:
        from robustmargin.empirical import ExactRates, per_cell_clopper_pearson

        cells = (
            Cell(severity=0.0, successes=200, trials=200),
            Cell(severity=0.01, successes=299, trials=300),
            Cell(severity=0.05, successes=122, trials=300),
        )

        rates = per_cell_clopper_pearson(cells, level=0.95)

        assert isinstance(rates, ExactRates)
        assert rates.severity == (0.0, 0.01, 0.05)
        assert rates.level == 0.95
        assert rates.method == "per_cell_clopper_pearson"
        assert len(rates.rate) == len(cells)
        assert len(rates.lo) == len(cells)
        assert len(rates.hi) == len(cells)

    def test_each_cells_rate_is_contained_in_its_own_exact_interval(self) -> None:
        from robustmargin.empirical import per_cell_clopper_pearson

        cells = (
            Cell(severity=0.0, successes=200, trials=200),
            Cell(severity=0.1, successes=0, trials=300),
        )

        rates = per_cell_clopper_pearson(cells, level=0.95)

        for lo, rate, hi in zip(rates.lo, rates.rate, rates.hi, strict=True):
            assert lo <= rate <= hi

    def test_all_success_cell_matches_the_two_sided_closed_form(self) -> None:
        """``per_cell_clopper_pearson`` is the **two-sided** exact interval at ``level``, which
        is what a reported per-cell rate is. For an all-success cell
        the closed form is ``(alpha/2) ** (1/n)`` with ``alpha = 1 - level``, so at ``level=0.95``
        the lower end is ``0.025 ** (1/500)``, not the one-sided ``0.05 ** (1/500)``.

        The one-sided closed forms are asserted against
        :func:`~robustmargin.censoring.exact_one_sided_bound`, above -- that function answers the
        one-sided question a censoring classification asks, and this one reports an interval.
        Keeping the two conventions apart is the whole point of having both.
        """
        from robustmargin.empirical import per_cell_clopper_pearson

        cells = (Cell(severity=10.0, successes=500, trials=500),)

        rates = per_cell_clopper_pearson(cells, level=0.95)

        assert rates.rate[0] == 1.0
        assert rates.lo[0] == pytest.approx(0.025 ** (1.0 / 500.0), rel=1e-12)
        assert rates.hi[0] == 1.0
        assert rates.level == 0.95

    def test_two_sided_interval_is_wider_than_the_one_sided_bound(self) -> None:
        """A regression guard for the convention itself: if someone re-wires
        ``per_cell_clopper_pearson`` back onto ``exact_one_sided_bound``, this fails.
        """
        from robustmargin.censoring import exact_one_sided_bound
        from robustmargin.empirical import per_cell_clopper_pearson

        cells = (Cell(severity=1.0, successes=98, trials=200),)
        rates = per_cell_clopper_pearson(cells, level=0.95)

        one_sided_lo = exact_one_sided_bound(98, 200, level=0.95, side="lower")
        one_sided_hi = exact_one_sided_bound(98, 200, level=0.95, side="upper")

        assert rates.lo[0] < one_sided_lo
        assert rates.hi[0] > one_sided_hi

    def test_accepts_fit_cells_directly(self) -> None:
        """The documented call shape: ``per_cell_clopper_pearson(fit.cells, ...)``, not
        ``per_cell_clopper_pearson(fit, ...)``.
        """
        from robustmargin.empirical import per_cell_clopper_pearson

        fit = fake_fit()

        rates = per_cell_clopper_pearson(fit.cells, level=0.95)

        assert rates.severity == tuple(c.severity for c in fit.cells)

    def test_default_level_is_0_95(self) -> None:
        from robustmargin.empirical import per_cell_clopper_pearson

        cells = (Cell(severity=0.0, successes=200, trials=200),)

        rates = per_cell_clopper_pearson(cells)

        assert rates.level == 0.95


def test_exact_rates_is_not_serialisable_through_report_to_dict() -> None:
    """Per-cell rates are labelled ``per_cell_clopper_pearson`` and never attached to a
    threshold. ``ExactRates`` is deliberately excluded from ``report.py``'s tagged classes and
    has no schema entry, so this is what structurally enforces that rule -- pinned here so a
    future change that makes ``ExactRates`` serialisable (and so makes it possible to smuggle a
    per-cell interval into a stored card next to a ``Threshold``) is caught.
    """
    from robustmargin.empirical import per_cell_clopper_pearson

    cells = (Cell(severity=0.0, successes=200, trials=200),)
    rates = per_cell_clopper_pearson(cells)

    with pytest.raises(TypeError):
        to_dict(rates)  # type: ignore[arg-type]


class TestOneSidedExactBoundFallback:
    """The exact-bound fallback's one-sided branch: reached when the bracket-level scan
    determines exactly one side.

    **A rescan at `level` must not reassign *both* sides.** If it did, a cell undetermined at
    `(1 + level) / 2` but failing at `level` would silently manufacture a second bound -- a
    two-sided bracket with each side at `level`, joint coverage about `2 * level - 1`, carried
    out to a `Threshold` still reporting `level`. That is exactly the mislabelling the
    bracket-level scan exists to prevent.

    The `(k, n)` pairs that expose it are those where `upper(level) < target <= upper((1+level)/2)`.
    `0/5` against a target of `0.5` at `level=0.95` is the smallest: `0.4507` at 0.95, `0.5218`
    at 0.975.
    """

    @staticmethod
    def _one_sided_fallback() -> Threshold:
        obs = Observations.from_counts(
            Axis(name="s", unit="u", scale="log"),
            severity=[1.0, 10.0],
            successes=[20, 0],
            trials=[20, 5],
            outcome="success",
            direction="decreasing",
        )
        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
        assert fit.status is Status.SEPARATION
        return threshold(
            fit, definition=Definition.absolute(0.5), interval_method="profile", level=0.95
        )

    def test_reports_exactly_one_bound_never_a_manufactured_bracket(self) -> None:
        t = self._one_sided_fallback()

        assert (t.lo is None) != (t.hi is None), (
            "a one-sided fallback must report exactly one bound; two bounds here would each be "
            "at `level`, making a bracket whose joint coverage is about 2*level - 1 while "
            "Threshold.level still says level"
        )
        assert t.value is None
        assert t.censoring is Censoring.NONE
        assert t.interval_method == "exact_bound"

    def test_the_warning_names_the_side_that_is_actually_set(self) -> None:
        """The side must be decided from the bracket-level scan, before the rescan. Computing it
        afterwards reported the opposite side, sending anyone debugging from the warning the
        wrong way.
        """
        t = self._one_sided_fallback()

        assert len(t.warnings) >= 1
        warning = next(w for w in t.warnings if "determined only the" in w)
        named_side = "lower" if "only the lower side" in warning else "upper"
        set_side = "lower" if t.lo is not None else "upper"
        assert named_side == set_side, f"warning says {named_side!r} but {set_side!r} is set"

    def test_the_bound_is_a_tested_level_and_sits_at_the_stated_confidence(self) -> None:
        t = self._one_sided_fallback()
        bound = t.lo if t.lo is not None else t.hi

        assert bound in (1.0, 10.0), "a fallback bound is always one of the tested severities"
        # The determined side here is the lower one: severity 1.0 is 20/20, whose one-sided
        # lower bound at 0.95 clears the 0.5 target. The point of the branch is that this is
        # computed at `level`, not at the bracket's `(1 + level) / 2`.
        assert t.lo == 1.0
        assert exact_one_sided_bound(20, 20, level=0.95, side="lower") >= 0.5
