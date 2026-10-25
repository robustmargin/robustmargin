"""``fit_dose_response`` on ``direction="increasing"``: the concrete mirror, the fit's own
messages, and the original-scale reading of everything it reports.

The generative version of the mirror is ``tests/property/test_direction_flip_properties.py``. This
file pins deterministic designs that are known to converge (so a failure to converge cannot make
a comparison vacuous), and the behaviours that are not numerical identities:

* ``Fit.direction``, ``Fit.cells`` (original counts, never complemented), ``Fit.monotonicity``
  (computed on the original ``Observations``) and ``cluster_ids``;
* ``CONTROL_INCOMPATIBLE`` for increasing data is a fixed ``lower=0`` over a control with events,
  and its message suggests ``lower="estimate"``, never ``upper="estimate"``;
* no warning on an increasing fit names a mirrored quantity;
* a fixed ``upper <= lower`` still raises, in both directions;
* aggregated and per-observation input still give the same fit.
"""

from __future__ import annotations

import itertools

import pytest

from robustmargin import (
    Axis,
    Fit,
    Observations,
    Status,
    check_monotonicity,
    fit_dose_response,
)

_LOG_AXIS = Axis(name="severity", unit="unit", scale="log")
_LINEAR_AXIS = Axis(name="severity", unit="unit", scale="linear")

# Events out of 400, rising from a nonzero floor (5%) to a ceiling below 1 (95%), with a control.
# Known to converge under every link with the asymptotes estimated, in the decreasing orientation
# that the mirror reduces to.
_SEVERITY = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
_EVENTS = [20, 22, 35, 90, 220, 340, 380]
_TRIALS = [400] * 7
_LINKS = ("probit", "logit", "cloglog")
_SWAP = {"mu": "mu", "s": "s", "upper": "lower", "lower": "upper"}


def _obs(direction: str, *, complement: bool, axis: Axis = _LOG_AXIS) -> Observations:
    successes = [n - k for k, n in zip(_EVENTS, _TRIALS, strict=True)] if complement else _EVENTS
    return Observations.from_counts(
        axis,
        severity=_SEVERITY,
        successes=successes,
        trials=_TRIALS,
        outcome="outcome",
        direction=direction,
    )


def _increasing(link: str, upper: float | str, lower: float | str) -> Fit:
    return fit_dose_response(
        _obs("increasing", complement=False), model="binomial", link=link, upper=upper, lower=lower
    )


def _mirrored(link: str, upper: float | str, lower: float | str) -> Fit:
    def flip(v: float | str) -> float | str:
        return v if isinstance(v, str) else 1.0 - v

    return fit_dose_response(
        _obs("decreasing", complement=True),
        model="binomial",
        link=link,
        upper=flip(lower),
        lower=flip(upper),
    )


def _cov(fit: Fit, a: str, b: str) -> float:
    assert fit.covariance is not None
    names = fit.covariance.names
    return fit.covariance.matrix[names.index(a)][names.index(b)]


# (upper, lower) on the outcome scale. The all-estimated and the fixed-inner cases both keep the
# control compatible; lower=0.0 with this control is the CONTROL_INCOMPATIBLE case tested below.
_OK_CASES = [("estimate", "estimate"), (1.0, "estimate"), (0.95, 0.05)]


class TestTheMirror:
    @pytest.mark.parametrize("link", _LINKS)
    @pytest.mark.parametrize(("upper", "lower"), _OK_CASES)
    def test_parameters_are_the_mirrored_decreasing_parameters(
        self, link: str, upper: float | str, lower: float | str
    ) -> None:
        inc = _increasing(link, upper, lower)
        dec = _mirrored(link, upper, lower)

        assert inc.status is Status.OK and dec.status is Status.OK
        assert inc.params is not None and dec.params is not None
        assert inc.direction == "increasing"
        for name in ("mu", "s"):
            assert inc.params[name].value == pytest.approx(dec.params[name].value, rel=1e-9)
        for name in ("upper", "lower"):
            assert inc.params[name].value == pytest.approx(
                1.0 - dec.params[_SWAP[name]].value, rel=1e-9, abs=1e-12
            )
            assert inc.params[name].fixed is dec.params[_SWAP[name]].fixed
        assert inc.params["lower"].value < inc.params["upper"].value
        assert inc.log_likelihood == pytest.approx(dec.log_likelihood, rel=1e-9)

    @pytest.mark.parametrize("link", _LINKS)
    @pytest.mark.parametrize(("upper", "lower"), _OK_CASES)
    def test_covariance_swaps_names_and_flips_the_sign_of_mixed_entries(
        self, link: str, upper: float | str, lower: float | str
    ) -> None:
        inc = _increasing(link, upper, lower)
        dec = _mirrored(link, upper, lower)
        assert inc.covariance is not None and dec.covariance is not None

        assert set(inc.covariance.names) == {_SWAP[n] for n in dec.covariance.names}
        for a, b in itertools.product(inc.covariance.names, repeat=2):
            flips = (a in ("upper", "lower")) != (b in ("upper", "lower"))
            expected = _cov(dec, _SWAP[a], _SWAP[b]) * (-1.0 if flips else 1.0)
            assert _cov(inc, a, b) == pytest.approx(expected, rel=1e-9, abs=1e-14)

    def test_a_mixed_entry_changes_sign_and_an_asymptote_pair_does_not(self) -> None:
        """With both asymptotes estimated, ``cov(mu, upper)`` of the increasing fit is minus
        ``cov(mu, lower')`` of the mirror, and ``cov(upper, lower)`` equals
        ``cov(lower', upper')`` unchanged."""
        inc = _increasing("probit", "estimate", "estimate")
        dec = _mirrored("probit", "estimate", "estimate")

        assert _cov(inc, "mu", "upper") == pytest.approx(-_cov(dec, "mu", "lower"), rel=1e-9)
        assert _cov(inc, "upper", "lower") == pytest.approx(_cov(dec, "lower", "upper"), rel=1e-9)
        assert _cov(inc, "mu", "upper") != 0.0

    @pytest.mark.parametrize("link", _LINKS)
    def test_diagnostics_are_those_of_the_mirrored_fit(self, link: str) -> None:
        inc = _increasing(link, "estimate", "estimate")
        dec = _mirrored(link, "estimate", "estimate")

        assert inc.diagnostics is not None and dec.diagnostics is not None
        assert inc.diagnostics.deviance == pytest.approx(dec.diagnostics.deviance, rel=1e-9)
        assert inc.diagnostics.dispersion == pytest.approx(dec.diagnostics.dispersion, rel=1e-9)
        assert inc.diagnostics.df == dec.diagnostics.df
        assert inc.diagnostics.link_aic.keys() == dec.diagnostics.link_aic.keys()
        for name, aic in dec.diagnostics.link_aic.items():
            other = inc.diagnostics.link_aic[name]
            if aic is None:
                assert other is None
            else:
                assert other == pytest.approx(aic, rel=1e-9)

    def test_cells_are_the_original_counts_never_the_complement(self) -> None:
        inc = _increasing("probit", "estimate", "estimate")

        assert [(c.severity, c.successes, c.trials) for c in inc.cells] == list(
            zip(_SEVERITY, _EVENTS, _TRIALS, strict=True)
        )

    def test_monotonicity_is_the_check_on_the_original_observations(self) -> None:
        inc = _increasing("probit", "estimate", "estimate")

        assert inc.monotonicity == check_monotonicity(_obs("increasing", complement=False))
        assert inc.monotonicity is not None
        assert inc.monotonicity.direction == "increasing"

    def test_the_fit_is_independent_of_the_aggregated_or_per_observation_form(self) -> None:
        severity: list[float] = []
        event: list[int] = []
        for s, k, n in zip(_SEVERITY, _EVENTS, _TRIALS, strict=True):
            severity += [s] * n
            event += [1] * k + [0] * (n - k)
        rows = Observations.from_observations(
            _LOG_AXIS, severity=severity, success=event, outcome="outcome", direction="increasing"
        )

        from_rows = fit_dose_response(
            rows, model="binomial", link="probit", upper="estimate", lower="estimate"
        )
        from_counts = _increasing("probit", "estimate", "estimate")

        assert from_rows.params == from_counts.params
        assert from_rows.log_likelihood == from_counts.log_likelihood

    def test_cluster_ids_come_from_the_original_observations(self) -> None:
        severity = [1.0] * 40 + [2.0] * 40 + [4.0] * 40
        event = [1] * 4 + [0] * 36 + [1] * 20 + [0] * 20 + [1] * 36 + [0] * 4
        cluster = ["a", "b"] * 60
        obs = Observations.from_observations(
            _LOG_AXIS,
            severity=severity,
            success=event,
            outcome="outcome",
            direction="increasing",
            cluster=cluster,
        )

        fit = fit_dose_response(
            obs, model="binomial", link="probit", upper=1.0, lower=0.0, dependence="independent"
        )

        assert fit.status is Status.OK
        assert fit.cluster_ids is not None
        assert set(fit.cluster_ids) == {"a", "b"}


class TestControlIncompatible:
    """A fixed ``lower=0`` says an event has probability exactly zero at the control, so any
    event there drives the likelihood to zero. The repair is to estimate ``lower``."""

    @pytest.mark.parametrize("upper", [1.0, "estimate"])
    def test_a_fixed_zero_lower_over_a_control_with_events_is_control_incompatible(
        self, upper: float | str
    ) -> None:
        fit = _increasing("probit", upper, 0.0)

        assert fit.status is Status.CONTROL_INCOMPATIBLE
        assert fit.params is None and fit.covariance is None and fit.log_likelihood is None

    def test_the_message_suggests_estimating_lower_and_never_upper(self) -> None:
        fit = _increasing("probit", 1.0, 0.0)

        assert any('lower="estimate"' in w for w in fit.warnings), fit.warnings
        assert not any('upper="estimate"' in w for w in fit.warnings), fit.warnings

    def test_a_control_without_events_is_compatible(self) -> None:
        obs = Observations.from_counts(
            _LOG_AXIS,
            severity=[0.0, 1.0, 2.0, 4.0, 8.0],
            successes=[0, 10, 40, 80, 95],
            trials=[100] * 5,
            outcome="outcome",
            direction="increasing",
        )

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        assert fit.status is Status.OK

    def test_a_linear_axis_has_no_zero_severity_control_to_be_incompatible_with(self) -> None:
        obs = Observations.from_counts(
            _LINEAR_AXIS,
            severity=[0.0, 1.0, 2.0, 4.0, 8.0],
            successes=[3, 10, 40, 80, 95],
            trials=[100] * 5,
            outcome="outcome",
            direction="increasing",
        )

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        assert fit.status is not Status.CONTROL_INCOMPATIBLE


class TestFailedFitsAreFlaggedNotNumbered:
    def test_events_that_fall_with_severity_do_not_converge_and_the_warning_names_no_mirror(
        self,
    ) -> None:
        """Events that fall as severity rises are the wrong sign for an increasing curve. The
        flag survives, and its text speaks of the direction the caller asked for."""
        obs = Observations.from_counts(
            _LOG_AXIS,
            severity=[0.0, 1.0, 2.0, 4.0, 8.0],
            successes=[40, 35, 25, 12, 5],
            trials=[100] * 5,
            outcome="outcome",
            direction="increasing",
        )

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower="estimate")

        assert fit.status is Status.NOT_CONVERGED
        assert fit.params is None and fit.log_likelihood is None
        assert fit.warnings
        assert not any("decreasing" in w for w in fit.warnings), fit.warnings

    def test_no_event_at_any_nonzero_severity_is_separation_worded_in_events(self) -> None:
        obs = Observations.from_counts(
            _LOG_AXIS,
            severity=[0.0, 1.0, 2.0, 4.0],
            successes=[0, 0, 0, 0],
            trials=[100] * 4,
            outcome="outcome",
            direction="increasing",
        )

        fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)

        assert fit.status is Status.SEPARATION
        assert any("no success was observed" in w for w in fit.warnings), fit.warnings
        assert not any("no failure" in w for w in fit.warnings), fit.warnings


class TestFixedAsymptoteValidation:
    @pytest.mark.parametrize("direction", ["decreasing", "increasing"])
    @pytest.mark.parametrize(("upper", "lower"), [(0.3, 0.3), (0.2, 0.4), (0.0, 0.0)])
    def test_a_fixed_upper_not_above_a_fixed_lower_raises_in_both_directions(
        self, direction: str, upper: float, lower: float
    ) -> None:
        obs = _obs(direction, complement=False)

        with pytest.raises(ValueError, match="upper"):
            fit_dose_response(obs, model="binomial", link="probit", upper=upper, lower=lower)

    def test_an_increasing_fit_no_longer_raises_for_the_direction_itself(self) -> None:
        fit = _increasing("probit", "estimate", "estimate")

        assert fit.direction == "increasing"
        assert fit.status is Status.OK
