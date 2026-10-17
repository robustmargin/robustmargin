"""Regression test for the generic (estimated-asymptote) fitting path's control-row
sensitivity.

With ``upper`` fixed at 1, adding an all-success zero-severity control row must leave a fit
unchanged: the control contributes exactly ``lchoose(n, n) + n*log(1) == 0`` to the full
binomial log-likelihood, for every parameter value, not only at the optimum.
``tests/property/test_models_properties.py``'s
``test_adding_an_all_success_control_leaves_a_fixed_u_1_fit_unchanged`` pins this down for the
``upper=1.0, lower=0.0`` (``GLM``) path. This file pins the same property for the
``upper=1.0, lower="estimate"`` (``GenericLikelihoodModel``) path, which a naive implementation
violates: statsmodels' own ``LikelihoodModel.fit`` minimises ``-loglike(params) / nobs`` (``nobs =
self.endog.shape[0]``, i.e. the number of *rows*, not pooled trials), so adding a
zero-contribution control row changes ``nobs`` and, with it, the scale at which ``gtol``/``tol``
decide convergence -- even though the log-likelihood *function itself* is unchanged at every
parameter value.

**Stated as invariance, not as an absolute "converges" assertion**, and **parametrized over all
three links, not one.** Both choices come from the same empirical finding, made by running this
exact fixture against an implementation without the correction, in three environments: this
development machine (macOS/x86_64, ``numpy==2.4.6 scipy==1.17.1 statsmodels==0.15.0``), a fresh
``python:3.11`` container (Linux/x86_64) at the same, independently-resolved "latest" versions, and
the same container against ``ci/constraints-min.txt``'s floors
(``numpy==1.26.0 scipy==1.11.1 statsmodels==0.14.0``):

==============  =======================  =======================  =======================
link            macOS, latest             Linux, latest             Linux, floor
==============  =======================  =======================  =======================
``probit``      NOT_CONVERGED either way  **OK -> NOT_CONVERGED**   **OK -> NOT_CONVERGED**
``logit``       **OK -> NOT_CONVERGED**   NOT_CONVERGED either way  **OK -> NOT_CONVERGED**
``cloglog``     OK -> OK (params agree)   OK -> OK (params agree)   OK -> OK (params agree)
==============  =======================  =======================  =======================

The defect is real in all three environments, but *which link's own "no control" fit happens to
sit on the converging side of this exact fixture's convergence boundary is itself
platform/dependency-version-dependent* -- an artifact of floating-point and BLAS differences
nudging an already-near-boundary optimisation, not of the defect changing shape. A test that
asserted a specific status for one hand-picked link (``without_control.status is Status.OK`` as
a "premise" for ``link="logit"`` alone) would report nothing at all on Linux at "latest":
``logit`` holds trivially there (both sides fail, for the unrelated boundary reason, saying
nothing about this defect), while ``probit`` is the link that actually demonstrates it there.

Parametrizing over all three links and asserting *status equality only* (never a specific side)
fixes this: in every environment checked so far, at least one of ``probit``/``logit`` lands on
the converging side without the control and so actually exercises the regression (both do, at
the dependency floor), while any link that does not holds vacuously (true, but uninformative)
instead of failing to report anything. ``cloglog`` is included for documentation -- it has not
been observed to trigger the defect in any environment checked, and its own parametrization is
expected to keep passing before and after a fix.
"""

from __future__ import annotations

import pytest

from robustmargin import Axis, Observations, Status, fit_dose_response

_AXIS = Axis(name="severity", unit="unit", scale="log")

# The exact counts the defect is demonstrated against: five well-separated log-spaced levels, trial
# counts of 50, and a success rate that falls smoothly from 48/50 to 3/50 -- not a knife-edge
# design by construction. Whether any *specific* link's own no-control fit happens to converge
# on it is still platform-dependent (see the module docstring's table), which is exactly why the
# test below asserts invariance across all three links rather than a specific status for one.
_SEVERITY = [1.0, 2.0, 4.0, 8.0, 16.0]
_SUCCESSES = [48, 40, 25, 12, 3]
_TRIALS = [50, 50, 50, 50, 50]
_CONTROL_TRIALS = 50


def _obs(*, with_control: bool) -> Observations:
    if with_control:
        severity = [0.0, *_SEVERITY]
        successes = [_CONTROL_TRIALS, *_SUCCESSES]
        trials = [_CONTROL_TRIALS, *_TRIALS]
    else:
        severity = list(_SEVERITY)
        successes = list(_SUCCESSES)
        trials = list(_TRIALS)
    return Observations.from_counts(
        _AXIS,
        severity=severity,
        successes=successes,
        trials=trials,
        outcome="success",
        direction="decreasing",
    )


class TestGenericPathControlRowInvariance:
    """``upper=1.0`` fixed, ``lower="estimate"`` -- the ``GenericLikelihoodModel`` path. All three
    links are parametrized (see the module docstring's cross-platform table): on any one platform,
    at most one or two of the three actually exercise the regression, but which one(s) do is not the
    same on every platform, so all three are kept rather than narrowed to whichever looked
    informative on a single machine.
    """

    @pytest.mark.parametrize("link", ["probit", "logit", "cloglog"])
    def test_status_and_converged_params_are_unchanged_by_an_all_success_control_row(
        self, link: str
    ) -> None:
        without_control = fit_dose_response(
            _obs(with_control=False), model="binomial", link=link, upper=1.0, lower="estimate"
        )
        with_control = fit_dose_response(
            _obs(with_control=True), model="binomial", link=link, upper=1.0, lower="estimate"
        )

        assert with_control.status == without_control.status, (
            "adding an all-success, zero-severity control row (which contributes exactly 0 to "
            "the log-likelihood under upper=1.0 fixed) must not change convergence status -- "
            f"got {without_control.status!r} without it and {with_control.status!r} with it "
            f"(link={link!r}, with_control warnings: {with_control.warnings!r})"
        )

        if without_control.status is not Status.OK:
            # A link that is vacuous on this platform (both sides already
            # fail, for the unrelated boundary reason the module docstring documents) still owes
            # evidence -- the two fits must fail for the *same* reason, not merely the same
            # status. Two different failure reasons behind one shared status would mean adding
            # the control row changed *something* about the fit even on the failing side, which
            # the status-only check above cannot see.
            assert with_control.warnings == without_control.warnings, (
                "both fits are non-OK with the same status, but fail for different reasons -- "
                f"without_control warnings {without_control.warnings!r}, with_control warnings "
                f"{with_control.warnings!r} (link={link!r})"
            )
            return

        assert without_control.params is not None and with_control.params is not None
        assert without_control.log_likelihood is not None
        assert with_control.log_likelihood is not None

        for name in ("mu", "s", "lower"):
            assert with_control.params[name].value == pytest.approx(
                without_control.params[name].value, rel=1e-6
            ), f"{name} differs between the two control-row conditions (link={link})"

        assert with_control.log_likelihood == pytest.approx(
            without_control.log_likelihood, rel=1e-6
        )
