"""Unit tests for ``fit_dose_response``: the GLM path and the estimated-asymptote path, the
"no interior maximum" rule, and the exact-overlap separation check.

Each status branch gets its own small, hand-constructed dataset rather than a real-world one,
so the test is about the *rule*, not about a particular series' numbers (the pooled-counts
fixture's edge cases are exercised separately in ``tests/unit/test_models_pooled_counts.py``).
One behaviour per test, named for what it asserts.

With u=1, l=0 the GLM path should agree with the likelihood path, but that path is not
separately reachable through the public API: there is no keyword to force the
``GenericLikelihoodModel`` path when both asymptotes are fixed at exactly 1.0/0.0, so that
equivalence cannot be expressed publicly and is not tested here.
"""

from __future__ import annotations

import pytest

from robustmargin import Axis, Observations, Status, fit_dose_response

_LOG_AXIS = Axis(name="severity", unit="unit", scale="log")


def _obs(
    *,
    severity: list[float],
    successes: list[int],
    trials: list[int],
    axis: Axis = _LOG_AXIS,
    direction: str = "decreasing",
    cluster: list[str] | None = None,
) -> Observations:
    return Observations.from_counts(
        axis,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction=direction,
        cluster=cluster,
    )


class TestWrongSignSlope:
    """Wrong-sign slope: performance rises with severity overall, with overlap (a success
    and a failure present at more than one level) and no separation -- the constrained maximum
    (``s > 0``) is the flat curve, so there is no finite ``(mu, s)``.
    """

    def test_returns_not_converged_with_no_numbers(self) -> None:
        # Success RATE rises with severity (10% -> 50% -> 90%): the opposite of what
        # direction="decreasing" declares, with every level carrying both successes and
        # failures (no separation).
        obs = _obs(
            severity=[0.1, 1.0, 10.0],
            successes=[10, 50, 90],
            trials=[100, 100, 100],
        )

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        assert fit.status is Status.NOT_CONVERGED
        assert fit.params is None
        assert fit.covariance is None
        assert fit.log_likelihood is None

    def test_warns_about_the_sign(self) -> None:
        obs = _obs(
            severity=[0.1, 1.0, 10.0],
            successes=[10, 50, 90],
            trials=[100, 100, 100],
        )

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        assert len(fit.warnings) > 0
        assert any("sign" in warning.lower() for warning in fit.warnings)

    @staticmethod
    def _non_exact_wrong_sign_obs() -> Observations:
        # Also rises with severity (12% -> 45% -> 91%), like the 10/50/90 case above, but this
        # one is not an exact probit fit (10/50/90 at these log-symmetric levels happens to be
        # reproduced exactly by a probit curve, which is why an exact fit is not evidence of
        # separation). This case confirms the wrong-sign check does not
        # depend on the fit being exact.
        return _obs(
            severity=[0.1, 1.0, 10.0],
            successes=[12, 45, 91],
            trials=[100, 100, 100],
        )

    def test_non_exact_wrong_sign_also_returns_not_converged(self) -> None:
        obs = self._non_exact_wrong_sign_obs()

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        assert fit.status is Status.NOT_CONVERGED
        assert fit.params is None
        assert fit.covariance is None
        assert fit.log_likelihood is None

    def test_non_exact_wrong_sign_warns_about_the_sign(self) -> None:
        obs = self._non_exact_wrong_sign_obs()

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        assert len(fit.warnings) > 0
        assert any("sign" in warning.lower() for warning in fit.warnings)

    def test_wrong_sign_with_upper_estimate_is_also_not_converged_with_sign_warning(self) -> None:
        """The wrong-sign check applies on the ``GenericLikelihoodModel`` path too, not
        just the GLM special case -- rising data with ``upper="estimate"`` must not be
        misreported as a boundary or separation issue; it is the same wrong-sign problem.
        """
        obs = _obs(
            severity=[0.1, 1.0, 10.0],
            successes=[20, 50, 80],
            trials=[100, 100, 100],
        )

        fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)

        assert fit.status is Status.NOT_CONVERGED
        assert fit.params is None
        assert fit.covariance is None
        assert fit.log_likelihood is None
        assert len(fit.warnings) > 0
        assert any("sign" in warning.lower() for warning in fit.warnings)


class TestEstimatedAsymptoteOnBoundary:
    """Boundary case: an estimated asymptote with a 400/400 control and a
    well-identified decreasing curve -- the control carries zero evidence against ``upper=1``,
    so the MLE for ``upper`` drifts to the boundary and the logit-scale parameter diverges.
    """

    @staticmethod
    def _obs_with_perfect_control() -> Observations:
        return _obs(
            severity=[0.0, 0.5, 1.0, 2.0, 4.0],
            successes=[400, 80, 40, 10, 2],
            trials=[400, 100, 100, 100, 100],
        )

    def test_returns_not_converged_with_no_numbers(self) -> None:
        obs = self._obs_with_perfect_control()

        fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)

        assert fit.status is Status.NOT_CONVERGED
        assert fit.params is None
        assert fit.covariance is None
        assert fit.log_likelihood is None

    def test_warns_about_the_boundary(self) -> None:
        obs = self._obs_with_perfect_control()

        fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)

        assert len(fit.warnings) > 0
        assert any("boundary" in warning.lower() for warning in fit.warnings)


class TestControlIncompatible:
    def test_fixed_upper_one_with_failing_control_is_control_incompatible(self) -> None:
        obs = _obs(
            severity=[0.0, 1.0, 2.0, 3.0],
            successes=[8, 90, 50, 10],
            trials=[10, 100, 100, 100],
        )

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        assert fit.status is Status.CONTROL_INCOMPATIBLE
        assert fit.params is None

    def test_warning_mentions_estimate(self) -> None:
        obs = _obs(
            severity=[0.0, 1.0, 2.0, 3.0],
            successes=[8, 90, 50, 10],
            trials=[10, 100, 100, 100],
        )

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        assert any("estimate" in warning.lower() for warning in fit.warnings)


class TestSeparation:
    def test_complete_separation_returns_separation_status(self) -> None:
        # Levels 1,2 are all-success (0 failures); levels 3,4 are all-failure (0 successes):
        # a perfect threshold exists between 2.0 and 3.0.
        obs = _obs(
            severity=[1.0, 2.0, 3.0, 4.0],
            successes=[100, 100, 0, 0],
            trials=[100, 100, 100, 100],
        )

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        assert fit.status is Status.SEPARATION
        assert fit.params is None
        assert fit.covariance is None
        assert fit.log_likelihood is None

    def test_two_overlapping_levels_is_ok_not_separation(self) -> None:
        """Regression test: statsmodels' ``PerfectSeparationWarning``
        tests ``allclose(fitted, observed)``, which also fires on an *exact* two-level binomial
        fit -- not just on genuine separation. Two overlapping levels (80/100 at severity 1.0,
        30/100 at severity 2.0) have a finite, unique MLE that a two-parameter probit
        reproduces exactly, so it must not be misreported as ``SEPARATION``: the separation
        decision rests on the exact pre-fit overlap check alone.
        """
        obs = _obs(
            severity=[1.0, 2.0],
            successes=[80, 30],
            trials=[100, 100],
        )

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        assert fit.status is Status.OK
        assert fit.params is not None


class TestDirectionIncreasingIsAccepted:
    def test_direction_increasing_returns_an_increasing_fit(self) -> None:
        obs = _obs(
            severity=[0.1, 1.0, 10.0],
            successes=[90, 50, 10],
            trials=[100, 100, 100],
            direction="increasing",
        )

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        assert fit.direction == "increasing"


class TestClusterIdsPassthrough:
    def test_fit_cluster_ids_equals_observations_unique_cluster_ids(self) -> None:
        severity = [0.1, 0.1, 1.0, 1.0, 10.0, 10.0]
        success = [1, 0, 1, 0, 0, 0]
        cluster = ["s1", "s2", "s1", "s2", "s1", "s2"]
        obs = Observations.from_observations(
            _LOG_AXIS,
            severity=severity,
            success=success,
            outcome="success",
            direction="decreasing",
            cluster=cluster,
        )

        # fit_dose_response threads dependence through to its own clustered-input guard (the
        # monotonicity check).
        fit = fit_dose_response(
            obs, model="binomial", link="probit", upper=1.0, lower=0.0, dependence="independent"
        )

        assert fit.cluster_ids is not None
        assert set(fit.cluster_ids) == set(obs.cluster or ())

    def test_fit_cluster_ids_is_none_when_observations_carry_no_clusters(self) -> None:
        obs = _obs(
            severity=[0.1, 1.0, 10.0],
            successes=[90, 50, 10],
            trials=[100, 100, 100],
        )

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        assert fit.cluster_ids is None


class TestFixedNonTrivialAsymptotes:
    """``upper=0.95`` (fixed, not estimated) is neither the GLM special case (``u=1, l=0``) nor
    an estimated asymptote -- it takes the ``GenericLikelihoodModel`` path with both asymptotes
    excluded from the parameter vector (fixed asymptotes are left out of the parameter vector).
    """

    def test_fixed_u_0_95_l_0_is_ok_with_mu_s_only_covariance(self) -> None:
        obs = _obs(
            severity=[0.0, 1.0, 2.0, 3.0, 4.0],
            successes=[94, 85, 60, 25, 8],
            trials=[100, 100, 100, 100, 100],
        )

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=0.95, lower=0.0)

        assert fit.status is Status.OK
        assert fit.params is not None
        assert fit.params["upper"].value == 0.95
        assert fit.params["upper"].fixed is True
        assert fit.params["lower"].value == 0.0
        assert fit.params["lower"].fixed is True
        assert fit.covariance is not None
        assert set(fit.covariance.names) == {"mu", "s"}


class TestFixedNonCanonicalUpperWithEstimatedLower:
    """``upper`` fixed at a non-canonical value (not ``1.0``) with ``lower="estimate"``
    must never raise, whatever the data looks like -- it either converges properly
    (``lower < upper``) or reports ``NOT_CONVERGED``, never an uncaught exception and never an
    invariant violation.
    """

    def test_rising_successes_with_upper_0_3_fixed_does_not_raise(self) -> None:
        # Successes RISE with severity (35% -> 45% -> 55% -> 65% -> 75%), the opposite of
        # direction="decreasing", with upper fixed below the data's own observed rates at the
        # high end -- a genuinely awkward combination for the optimizer.
        obs = _obs(
            severity=[0.1, 1.0, 10.0, 100.0, 1000.0],
            successes=[70, 90, 110, 130, 150],
            trials=[200, 200, 200, 200, 200],
        )

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=0.3, lower="estimate")

        assert fit.status in (Status.NOT_CONVERGED, Status.OK)
        if fit.status is Status.OK:
            assert fit.params is not None
            assert fit.params["lower"].value < fit.params["upper"].value

    def test_normal_decreasing_design_with_upper_0_9_fixed_is_ok(self) -> None:
        obs = _obs(
            severity=[0.1, 1.0, 10.0, 100.0, 1000.0],
            successes=[176, 160, 90, 25, 5],
            trials=[200, 200, 200, 200, 200],
        )

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=0.9, lower="estimate")

        assert fit.status is Status.OK
        assert fit.params is not None
        assert fit.params["upper"].value == 0.9
        assert fit.params["upper"].fixed is True
        assert 0.0 <= fit.params["lower"].value < 0.9


class TestExactStepWithEstimatedAsymptotes:
    """An exact step -- three identical high rates, then three identical low rates, plus a
    control matching the high rate -- gives the estimated-asymptote path almost no information
    about the transition's shape (``s``) or location beyond "somewhere between the third and
    fourth nonzero level". This must report ``NOT_CONVERGED`` for the step itself and must never
    raise, for any link.
    """

    @staticmethod
    def _obs_with_exact_step() -> Observations:
        return _obs(
            severity=[0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
            successes=[190, 190, 190, 190, 10, 10, 10],
            trials=[200, 200, 200, 200, 200, 200, 200],
        )

    @pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
    def test_is_not_converged_as_a_step_between_adjacent_levels(self, link: str) -> None:
        obs = self._obs_with_exact_step()

        fit = fit_dose_response(
            obs, model="binomial", link=link, upper="estimate", lower="estimate"
        )

        assert fit.status is Status.NOT_CONVERGED
        assert fit.params is None
        assert fit.covariance is None
        assert fit.log_likelihood is None
        assert any("a step between adjacent tested levels" in w for w in fit.warnings)
