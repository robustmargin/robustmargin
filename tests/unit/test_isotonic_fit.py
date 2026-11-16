"""``fit_dose_response(model="isotonic")``: the monotone nonparametric fit.

An isotonic fit carries no parameters and no link. Its content is ``cell_estimates``: the
weighted least-squares monotone fit to the cell rates (counts, weights ``trials``) or cell means
(continuous, weights ``n``), computed by pool-adjacent-violators, one value per cell in ``cells``
order, the control included. Its ``log_likelihood`` is the saturated-at-the-estimates value:

* counts: ``sum(log C(n, k) + k * log(p) + (n - k) * log(1 - p))`` at ``p = cell_estimates``,
  with ``0 * log(0) = 0``; the binomial coefficient is included, as in the parametric binomial
  fits, so the value is comparable to theirs;
* continuous: the Gaussian log-likelihood with ``sigma2 = RSS / N`` at its maximum,
  ``-N / 2 * (log(2 * pi * sigma2) + 1)``, where
  ``RSS = sum((n_i - 1) * sd_i**2) + sum(n_i * (mean_i - estimate_i)**2)``.

``predict`` and ``diagnose`` need a parametric curve, so they refuse. Every other model has
``cell_estimates is None``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import gammaln

from robustmargin import (
    Axis,
    ContinuousObservations,
    Observations,
    Status,
    diagnose,
    fit_dose_response,
)

_LOG = Axis(name="severity", unit="unit", scale="log")
_LINEAR = Axis(name="severity", unit="unit", scale="linear")
_SEVERITY = [0.0, 1.0, 2.0, 4.0, 8.0]


def _log_choose(n: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.asarray(gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0))


def _counts(
    successes: list[int],
    trials: list[int],
    *,
    direction: str = "decreasing",
    axis: Axis = _LOG,
    cluster: list[str] | None = None,
) -> Observations:
    return Observations.from_counts(
        axis,
        severity=_SEVERITY[: len(successes)],
        successes=successes,
        trials=trials,
        outcome="outcome",
        direction=direction,
        cluster=cluster,
    )


def _continuous(
    mean: list[float],
    n: list[int],
    *,
    sd: list[float] | None = None,
    direction: str = "decreasing",
    axis: Axis = _LOG,
) -> ContinuousObservations:
    return ContinuousObservations.from_summaries(
        axis,
        severity=_SEVERITY[: len(mean)],
        mean=mean,
        sd=sd if sd is not None else [1.0] * len(mean),
        n=n,
        outcome="response",
        direction=direction,
    )


# Weighted PAVA by hand (decreasing: a block is pooled with its neighbour while it is *below*
# the next block).
#   rates      .98  .80  .90  .40  .45     trials 100 50 150 100 100
#   .80 < .90 violates: pool 40 + 135 = 175 successes in 200 trials -> .875
#   .40 < .45 violates: pool 40 + 45 = 85 in 200 -> .425
#   .98 >= .875 >= .425, so the blocks stand.
_DEC_SUCCESSES = [98, 40, 135, 40, 45]
_DEC_TRIALS = [100, 50, 150, 100, 100]
_DEC_ESTIMATES = [0.98, 0.875, 0.875, 0.425, 0.425]

# Increasing: a block is pooled with its neighbour while it is *above* the next block.
#   rates      .02  .10  .07  .60  .55     trials 100 each
#   .10 > .07 violates: (10 + 7) / 200 = .085;  .60 > .55 violates: (60 + 55) / 200 = .575.
_INC_SUCCESSES = [2, 10, 7, 60, 55]
_INC_TRIALS = [100] * 5
_INC_ESTIMATES = [0.02, 0.085, 0.085, 0.575, 0.575]

# Continuous, decreasing, weights n:  means 10  8  9  3  4,  n 4 2 6 4 4
#   8 (n=2) < 9 (n=6): (16 + 54) / 8 = 8.75;  3 (n=4) < 4 (n=4): 3.5.
_CONT_MEAN = [10.0, 8.0, 9.0, 3.0, 4.0]
_CONT_N = [4, 2, 6, 4, 4]
_CONT_ESTIMATES = [10.0, 8.75, 8.75, 3.5, 3.5]


class TestFitShape:
    def test_counts_fit_carries_no_parameters_and_no_link(self) -> None:
        fit = fit_dose_response(_counts(_DEC_SUCCESSES, _DEC_TRIALS), model="isotonic")

        assert fit.model == "isotonic"
        assert fit.link == "none"
        assert fit.status is Status.OK
        assert fit.params is None
        assert fit.covariance is None
        assert fit.diagnostics is None
        assert fit.residual is None
        assert isinstance(fit.log_likelihood, float)

    def test_continuous_fit_carries_no_parameters_and_no_link(self) -> None:
        fit = fit_dose_response(_continuous(_CONT_MEAN, _CONT_N), model="isotonic")

        assert fit.model == "isotonic"
        assert fit.link == "none"
        assert fit.status is Status.OK
        assert (fit.params, fit.covariance, fit.diagnostics, fit.residual) == (None,) * 4
        assert isinstance(fit.log_likelihood, float)

    def test_monotonicity_is_computed_as_for_the_other_models(self) -> None:
        fit = fit_dose_response(_counts(_DEC_SUCCESSES, _DEC_TRIALS), model="isotonic")

        assert fit.monotonicity is not None
        assert fit.monotonicity.direction == "decreasing"

    def test_the_fit_keeps_axis_outcome_direction_and_cells(self) -> None:
        obs = _counts(_INC_SUCCESSES, _INC_TRIALS, direction="increasing")

        fit = fit_dose_response(obs, model="isotonic")

        assert fit.axis == obs.axis and fit.outcome == "outcome"
        assert fit.direction == "increasing"
        assert fit.cells == obs.cells()

    def test_the_fit_is_a_frozen_value_that_equals_a_refit(self) -> None:
        obs = _counts(_DEC_SUCCESSES, _DEC_TRIALS)

        assert fit_dose_response(obs, model="isotonic") == fit_dose_response(obs, model="isotonic")


class TestCellEstimates:
    def test_decreasing_counts_match_hand_computed_weighted_pava_with_the_control(self) -> None:
        fit = fit_dose_response(_counts(_DEC_SUCCESSES, _DEC_TRIALS), model="isotonic")

        assert fit.cell_estimates is not None
        assert len(fit.cell_estimates) == len(fit.cells) == 5
        assert fit.cell_estimates == pytest.approx(_DEC_ESTIMATES, rel=1e-12)

    def test_increasing_counts_match_hand_computed_weighted_pava(self) -> None:
        fit = fit_dose_response(
            _counts(_INC_SUCCESSES, _INC_TRIALS, direction="increasing"), model="isotonic"
        )

        assert fit.cell_estimates == pytest.approx(_INC_ESTIMATES, rel=1e-12)

    def test_decreasing_continuous_match_hand_computed_weighted_pava(self) -> None:
        fit = fit_dose_response(_continuous(_CONT_MEAN, _CONT_N), model="isotonic")

        assert fit.cell_estimates == pytest.approx(_CONT_ESTIMATES, rel=1e-12)

    def test_increasing_continuous_is_pava_on_the_nondecreasing_side(self) -> None:
        # means 1 3 2 2 9 with n 2 2 2 6 2: 3 > 2 violates -> 2.5 (n=4); then 2.5 > 2 (n=6)
        # violates -> (10 + 12) / 10 = 2.2.
        fit = fit_dose_response(
            _continuous([1.0, 3.0, 2.0, 2.0, 9.0], [2, 2, 2, 6, 2], direction="increasing"),
            model="isotonic",
        )

        assert fit.cell_estimates == pytest.approx([1.0, 2.2, 2.2, 2.2, 9.0], rel=1e-12)

    def test_the_estimates_are_on_the_outcome_as_counted_not_the_complement(self) -> None:
        fit = fit_dose_response(
            _counts(_INC_SUCCESSES, _INC_TRIALS, direction="increasing"), model="isotonic"
        )

        assert fit.cell_estimates is not None
        assert fit.cell_estimates[0] == pytest.approx(0.02)
        assert fit.cell_estimates[-1] == pytest.approx(0.575)

    def test_already_monotone_data_is_returned_unchanged(self) -> None:
        fit = fit_dose_response(_counts([100, 90, 60, 20, 5], [100] * 5), model="isotonic")

        assert fit.cell_estimates == pytest.approx([1.0, 0.9, 0.6, 0.2, 0.05], rel=1e-12)

    def test_a_fully_reversed_series_pools_to_one_weighted_mean(self) -> None:
        # Decreasing outcome whose rates all rise: one block, the pooled rate (2 + 3 + 4) / 90.
        fit = fit_dose_response(_counts([2, 3, 4], [30, 30, 30]), model="isotonic")

        assert fit.cell_estimates == pytest.approx([0.1] * 3, rel=1e-12)

    def test_the_estimates_are_monotone_in_the_fitted_direction(self) -> None:
        dec = fit_dose_response(_counts(_DEC_SUCCESSES, _DEC_TRIALS), model="isotonic")
        inc = fit_dose_response(
            _counts(_INC_SUCCESSES, _INC_TRIALS, direction="increasing"), model="isotonic"
        )

        assert dec.cell_estimates is not None and inc.cell_estimates is not None
        assert np.all(np.diff(dec.cell_estimates) <= 0.0)
        assert np.all(np.diff(inc.cell_estimates) >= 0.0)

    def test_a_linear_axis_control_is_a_cell_like_any_other(self) -> None:
        fit = fit_dose_response(
            _counts(_DEC_SUCCESSES, _DEC_TRIALS, axis=_LINEAR), model="isotonic"
        )

        assert fit.cell_estimates == pytest.approx(_DEC_ESTIMATES, rel=1e-12)

    def test_the_estimates_are_a_tuple_of_floats(self) -> None:
        fit = fit_dose_response(_counts(_DEC_SUCCESSES, _DEC_TRIALS), model="isotonic")

        assert isinstance(fit.cell_estimates, tuple)
        assert all(isinstance(v, float) for v in fit.cell_estimates)

    @pytest.mark.parametrize("model", ["binomial", "gaussian"])
    def test_cell_estimates_is_none_on_a_parametric_fit(self, model: str) -> None:
        if model == "binomial":
            fit = fit_dose_response(
                _counts([100, 95, 70, 30, 5], [100] * 5),
                model="binomial",
                link="probit",
                upper=1.0,
                lower=0.0,
            )
        else:
            rng = np.random.default_rng(3)
            severity = [x for x in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0] for _ in range(4)]
            value = [
                10.0 - 9.0 / (1.0 + math.exp(-(math.log(x) - 0.5) / 0.4)) if x > 0 else 10.0
                for x in severity
            ]
            obs = ContinuousObservations.from_observations(
                _LOG,
                severity=severity,
                value=[v + float(rng.normal(0.0, 0.2)) for v in value],
                outcome="response",
                direction="decreasing",
            )
            fit = fit_dose_response(
                obs, model="gaussian", link="logit", upper="estimate", lower="estimate"
            )

        assert fit.status is Status.OK
        assert fit.cell_estimates is None


class TestLogLikelihood:
    def test_binomial_value_matches_a_hand_computed_case(self) -> None:
        # Estimates [1, .5, .2]: 10 ln(.5) (control: 0, then 5 ln .5 + 5 ln .5)
        # + 2 ln(.2) + 8 ln(.8) = -6.931472 - 3.218876 - 1.785148 = -11.935496, plus
        # ln C(10, 10) + ln C(10, 5) + ln C(10, 2) = 0 + 5.529429 + 3.806662.
        fit = fit_dose_response(_counts([10, 5, 2], [10, 10, 10]), model="isotonic")

        assert fit.log_likelihood == pytest.approx(-2.599405, abs=1e-6)

    def test_binomial_uses_the_pooled_estimates_not_the_raw_rates(self) -> None:
        fit = fit_dose_response(_counts(_DEC_SUCCESSES, _DEC_TRIALS), model="isotonic")

        p = np.asarray(_DEC_ESTIMATES)
        k = np.asarray(_DEC_SUCCESSES, dtype=float)
        n = np.asarray(_DEC_TRIALS, dtype=float)
        expected = float(np.sum(_log_choose(n, k) + k * np.log(p) + (n - k) * np.log1p(-p)))
        assert fit.log_likelihood == pytest.approx(expected, rel=1e-12)

    def test_binomial_includes_the_binomial_coefficient(self) -> None:
        # One cell of 1 success in 2 trials at rate .5 carries ln C(2, 1) = ln 2 on top of
        # 2 ln(.5), so the total is ln(.5). Three cells keep the fit well posed.
        fit = fit_dose_response(_counts([2, 1, 0], [2, 2, 2]), model="isotonic")

        assert fit.log_likelihood == pytest.approx(math.log(0.5), rel=1e-12)

    def test_binomial_zero_times_log_zero_is_zero(self) -> None:
        # Rates 1, 1, 0: every cell sits on 0 or 1, so every term is 0 * log(0) = 0.
        fit = fit_dose_response(_counts([10, 10, 0], [10, 10, 10]), model="isotonic")

        assert fit.log_likelihood == pytest.approx(0.0, abs=1e-12)

    def test_binomial_increasing_is_the_same_formula_on_the_counted_outcome(self) -> None:
        fit = fit_dose_response(
            _counts(_INC_SUCCESSES, _INC_TRIALS, direction="increasing"), model="isotonic"
        )

        p = np.asarray(_INC_ESTIMATES)
        k = np.asarray(_INC_SUCCESSES, dtype=float)
        n = np.asarray(_INC_TRIALS, dtype=float)
        expected = float(np.sum(_log_choose(n, k) + k * np.log(p) + (n - k) * np.log1p(-p)))
        assert fit.log_likelihood == pytest.approx(expected, rel=1e-12)

    def test_continuous_value_is_the_gaussian_log_likelihood_at_the_mle_variance(self) -> None:
        sd = [0.5, 1.0, 1.5, 0.8, 1.2]
        fit = fit_dose_response(_continuous(_CONT_MEAN, _CONT_N, sd=sd), model="isotonic")

        n = np.asarray(_CONT_N, dtype=float)
        mean = np.asarray(_CONT_MEAN)
        est = np.asarray(_CONT_ESTIMATES)
        rss = float(np.sum((n - 1.0) * np.asarray(sd) ** 2) + np.sum(n * (mean - est) ** 2))
        total = float(n.sum())
        sigma2 = rss / total
        expected = -total / 2.0 * (math.log(2.0 * math.pi * sigma2) + 1.0)
        assert fit.log_likelihood == pytest.approx(expected, rel=1e-12)

    def test_continuous_value_matches_a_hand_computed_case(self) -> None:
        # Cells (mean, sd, n): (4, 1, 2), (2, 1, 2), (3, 1, 2) decreasing. 2 < 3 violates:
        # pooled 2.5. RSS = 3 * (2 - 1) * 1 + 2 * (4 - 4)**2 + 2 * (2 - 2.5)**2 + 2 * (3 - 2.5)**2
        # = 3 + 0 + .5 + .5 = 4; N = 6; sigma2 = 2/3; ll = -3 * (ln(2 pi 2/3) + 1) = -7.2972...
        fit = fit_dose_response(
            ContinuousObservations.from_summaries(
                _LOG,
                severity=[0.0, 1.0, 2.0],
                mean=[4.0, 2.0, 3.0],
                sd=[1.0, 1.0, 1.0],
                n=[2, 2, 2],
                outcome="response",
                direction="decreasing",
            ),
            model="isotonic",
        )

        expected = -3.0 * (math.log(2.0 * math.pi * 2.0 / 3.0) + 1.0)
        assert fit.log_likelihood == pytest.approx(expected, rel=1e-12)
        assert fit.log_likelihood == pytest.approx(-7.29724, abs=1e-4)


class TestArguments:
    @pytest.mark.parametrize(
        "extra", [{"link": "probit"}, {"upper": 1.0}, {"lower": 0.0}, {"upper": "estimate"}]
    )
    def test_a_parametric_argument_with_isotonic_raises_value_error(
        self, extra: dict[str, object]
    ) -> None:
        with pytest.raises(ValueError):
            fit_dose_response(
                _counts(_DEC_SUCCESSES, _DEC_TRIALS),
                model="isotonic",
                **extra,  # type: ignore[call-overload]
            )

    def test_all_three_parametric_arguments_with_isotonic_raise_value_error(self) -> None:
        with pytest.raises(ValueError):
            fit_dose_response(
                _continuous(_CONT_MEAN, _CONT_N),
                model="isotonic",
                link="logit",  # type: ignore[call-overload]
                upper=1.0,
                lower=0.0,
            )

    def test_the_value_error_names_the_argument_that_does_not_belong(self) -> None:
        with pytest.raises(ValueError, match="link"):
            fit_dose_response(
                _counts(_DEC_SUCCESSES, _DEC_TRIALS),
                model="isotonic",
                link="probit",  # type: ignore[call-overload]
            )

    def test_a_binomial_fit_without_link_raises_type_error_naming_it(self) -> None:
        with pytest.raises(TypeError, match="link"):
            fit_dose_response(
                _counts(_DEC_SUCCESSES, _DEC_TRIALS),
                model="binomial",  # type: ignore[call-overload]
                upper=1.0,
                lower=0.0,
            )

    @pytest.mark.parametrize("missing", ["upper", "lower"])
    def test_a_binomial_fit_without_an_asymptote_raises_type_error_naming_it(
        self, missing: str
    ) -> None:
        kwargs: dict[str, object] = {"link": "probit", "upper": 1.0, "lower": 0.0}
        del kwargs[missing]

        with pytest.raises(TypeError, match=missing):
            fit_dose_response(
                _counts(_DEC_SUCCESSES, _DEC_TRIALS),
                model="binomial",
                **kwargs,  # type: ignore[call-overload]
            )

    def test_a_gaussian_fit_without_link_raises_type_error_naming_it(self) -> None:
        with pytest.raises(TypeError, match="link"):
            fit_dose_response(
                _continuous(_CONT_MEAN, _CONT_N),
                model="gaussian",  # type: ignore[call-overload]
                upper="estimate",
                lower="estimate",
            )


class TestDependenceAndClusters:
    _CLUSTER = ["a", "a", "b", "b", "c"]

    def test_clustered_input_without_dependence_raises(self) -> None:
        obs = _counts(_DEC_SUCCESSES, _DEC_TRIALS, cluster=self._CLUSTER)

        with pytest.raises(ValueError):
            fit_dose_response(obs, model="isotonic")

    def test_clustered_input_with_explicit_independence_fits_and_keeps_cluster_ids(self) -> None:
        obs = _counts(_DEC_SUCCESSES, _DEC_TRIALS, cluster=self._CLUSTER)

        fit = fit_dose_response(obs, model="isotonic", dependence="independent")

        assert fit.status is Status.OK
        assert fit.cluster_ids == ("a", "b", "c")

    def test_a_dependence_other_than_independent_raises(self) -> None:
        with pytest.raises(ValueError):
            fit_dose_response(
                _counts(_DEC_SUCCESSES, _DEC_TRIALS), model="isotonic", dependence="paired"
            )

    def test_unclustered_input_has_no_cluster_ids(self) -> None:
        fit = fit_dose_response(_counts(_DEC_SUCCESSES, _DEC_TRIALS), model="isotonic")

        assert fit.cluster_ids is None


class TestPredictAndDiagnoseRefuse:
    @pytest.mark.parametrize("which", ["counts", "continuous"])
    def test_predict_raises_value_error_mentioning_cell_estimates(self, which: str) -> None:
        obs = (
            _counts(_DEC_SUCCESSES, _DEC_TRIALS)
            if which == "counts"
            else _continuous(_CONT_MEAN, _CONT_N)
        )
        fit = fit_dose_response(obs, model="isotonic")

        with pytest.raises(ValueError, match="cell_estimates"):
            fit.predict(1.5)

    @pytest.mark.parametrize("which", ["counts", "continuous"])
    def test_diagnose_raises_value_error(self, which: str) -> None:
        obs = (
            _counts(_DEC_SUCCESSES, _DEC_TRIALS)
            if which == "counts"
            else _continuous(_CONT_MEAN, _CONT_N)
        )
        fit = fit_dose_response(obs, model="isotonic")

        with pytest.raises(ValueError):
            diagnose(fit)
