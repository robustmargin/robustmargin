"""A replacement start's polish may fall only within the accepted tolerance below its BFGS value.

When BFGS from an alternative start, or from the log-likelihood scan, beats the selected result
by more than the optimality tolerance (1e-6), it replaces that result and is Newton-polished. The
fit then refuses (``NOT_CONVERGED``) if that polish falls more than ``1e-8 * (1 + |llf|)`` below the
replacement's own BFGS log-likelihood ``B``: the tolerance a single-start fit accepts. So the
reported ``OK`` point is never more than that tolerance below the best value any stage reached, and
a larger drop is refused rather than silently swapped for the lower, replaced result.

The tests make this deterministic. A recording run finds the BFGS value ``B`` that feeds the
replacement's polish; the checked run then reports that polish's log-likelihood as chosen.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from statsmodels.base.model import GenericLikelihoodModel

from robustmargin import Axis, Observations, Status, fit_dose_response
from robustmargin import models as models_module

_LOG = Axis(name="severity", unit="unit", scale="log")
_LINEAR = Axis(name="severity", unit="unit", scale="linear")

_WITHIN_TOLERANCE_BELOW_BFGS = 2.0e-6


class _Reported:
    """A statsmodels result with its reported log-likelihood replaced."""

    def __init__(self, inner: Any, llf: float) -> None:
        self._inner = inner
        self.llf = llf

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


_GAP_SEVERITY = [
    0.0,
    15.0,
    37.7879274206032,
    58.9841931089883,
    59.3788140504317,
    84.39271170536418,
]
_GAP_SUCCESSES = [1, 26, 14, 27, 1, 12]
_GAP_TRIALS = [10, 49, 26, 47, 20, 14]
_R21_SEVERITY = [54.966, 84.166, 95.205, 95.73]
_R21_SUCCESSES = [6, 6, 6, 3]


def _scan_design() -> tuple[Observations, dict[str, Any], bool]:
    """Pilot, then a scan start that beats it; alternative starts are disabled in every run.

    Where the pilot lands differs across platforms, so alternative starts are switched off and
    the only possible replacement is the scan's, at polish index 1.
    """
    obs = Observations.from_counts(
        _LOG,
        severity=_GAP_SEVERITY,
        successes=[k * 8 for k in _GAP_SUCCESSES],
        trials=[n * 8 for n in _GAP_TRIALS],
        outcome="success",
        direction="decreasing",
    )
    return obs, {"link": "cloglog", "upper": 0.95, "lower": "estimate"}, True


def _alternative_design() -> tuple[Observations, dict[str, Any], bool]:
    """Pilot in a local maximum, replaced by an alternative start; the scan finds nothing."""
    obs = Observations.from_counts(
        _LINEAR,
        severity=_R21_SEVERITY,
        successes=[k * 200 for k in _R21_SUCCESSES],
        trials=[2000] * 4,
        outcome="success",
        direction="decreasing",
    )
    return obs, {"link": "logit", "upper": 0.95, "lower": 0.05}, False


def _fit_first_call_only(
    monkeypatch: pytest.MonkeyPatch,
    design: tuple[Observations, dict[str, Any], bool],
    report: dict[int, float] | None,
) -> tuple[Any, list[float]]:
    """Fit, intercepting the Newton polishes of the first direct-likelihood call.

    Returns the fit and the BFGS log-likelihood that fed each Newton polish, in order. With
    ``report`` given, polish number ``i`` reports ``report[i]`` as its log-likelihood.
    """
    real_fit = GenericLikelihoodModel.fit
    real_generic = models_module._fit_generic
    bfgs_llf_by_params: dict[bytes, float] = {}
    newton_inputs: list[float] = []
    state = {"active": False, "done": False}

    def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = real_fit(self, *args, **kwargs)
        if not state["active"]:
            return result
        if kwargs.get("method") == "bfgs":
            key = np.asarray(result.params, dtype=np.float64).tobytes()
            bfgs_llf_by_params[key] = float(result.llf)
            return result
        key = np.asarray(kwargs["start_params"], dtype=np.float64).tobytes()
        newton_inputs.append(bfgs_llf_by_params[key])
        index = len(newton_inputs) - 1
        if report is not None and index in report:
            return _Reported(result, report[index])
        return result

    def generic(*args: Any, **kwargs: Any) -> Any:
        if state["done"]:
            return real_generic(*args, **kwargs)
        state["active"] = True
        try:
            return real_generic(*args, **kwargs)
        finally:
            state["active"] = False
            state["done"] = True

    monkeypatch.setattr(GenericLikelihoodModel, "fit", spy)
    monkeypatch.setattr(models_module, "_fit_generic", generic)
    obs, kwargs, scan_design = design
    if scan_design:
        monkeypatch.setattr(models_module, "_ALTERNATIVE_STARTS", ())
    fitted = fit_dose_response(obs, model="binomial", **kwargs)
    return fitted, newton_inputs


def _recorded(
    monkeypatch: pytest.MonkeyPatch, design: tuple[Observations, dict[str, Any], bool]
) -> tuple[Any, list[float]]:
    with monkeypatch.context() as patch:
        return _fit_first_call_only(patch, design, None)


def _recorded_bfgs_values(
    monkeypatch: pytest.MonkeyPatch, design: tuple[Observations, dict[str, Any], bool]
) -> list[float]:
    return _recorded(monkeypatch, design)[1]


def test_scan_design_has_a_scan_start_that_replaces_the_pilot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Polish 0 is the pilot's, 1 the scan's. A platform whose pilot already lands in the scan's
    # basin leaves no replacement to test; that is a failure of this premise, not a skip.
    inputs = _recorded_bfgs_values(monkeypatch, _scan_design())

    assert len(inputs) == 2, (
        f"expected the scan to replace the pilot; got polish inputs {inputs} "
        "(the pilot may have landed in the scan's basin on this platform)"
    )
    assert inputs[1] > inputs[0] + 1e-6
    assert abs(inputs[1]) > 200.0


def test_alternative_design_replaces_the_pilot_once(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _recorded_bfgs_values(monkeypatch, _alternative_design())

    assert len(inputs) == 2
    assert inputs[1] > inputs[0] + 1e-6
    assert abs(inputs[1]) > 200.0


_DESIGNS = {
    "alternative_start": (_alternative_design, 1, False),
    "scan": (_scan_design, 1, True),
}


def _fit_with_replacement_polish_reported(
    monkeypatch: pytest.MonkeyPatch, name: str, below_bfgs: float
) -> tuple[Any, Any, float]:
    """The unpatched fit, and the fit whose replacement polish reports ``B - below_bfgs``."""
    make_design, replacement, scan_enabled = _DESIGNS[name]
    design = make_design()
    unpatched, inputs = _recorded(monkeypatch, design)
    bfgs_value = inputs[replacement]
    with monkeypatch.context() as patch:
        if not scan_enabled:
            # Leave no scan replacement to hide the one under test. (The scan design disables
            # alternative starts inside the fit helper, in every run.)
            patch.setattr(models_module, "_scan_starts", lambda *args, **kwargs: [])
        fitted, _ = _fit_first_call_only(patch, design, {replacement: bfgs_value - below_bfgs})
    return unpatched, fitted, bfgs_value


@pytest.mark.parametrize("name", list(_DESIGNS))
def test_replacement_polished_slightly_below_its_bfgs_value_is_kept_as_ok(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    unpatched, fitted, bfgs_value = _fit_with_replacement_polish_reported(
        monkeypatch, name, _WITHIN_TOLERANCE_BELOW_BFGS
    )

    # The polish sits below the replaced result's BFGS value by less than the gap that triggered
    # the replacement, but within the tolerance a single-start fit accepts for its own polish.
    assert _WITHIN_TOLERANCE_BELOW_BFGS < 1e-8 * (1.0 + abs(bfgs_value))
    assert fitted.status is Status.OK
    # The replacement's parameters, not the replaced result's, are what is reported.
    assert math.isclose(
        float(fitted.params["mu"].value), float(unpatched.params["mu"].value), rel_tol=1e-9
    )


@pytest.mark.parametrize("below_bfgs", [1e-3, 5.0, math.inf, math.nan])
@pytest.mark.parametrize("name", list(_DESIGNS))
def test_replacement_polished_far_below_its_bfgs_value_is_not_converged_never_ok(
    monkeypatch: pytest.MonkeyPatch, name: str, below_bfgs: float
) -> None:
    _, fitted, _ = _fit_with_replacement_polish_reported(monkeypatch, name, below_bfgs)

    # Refused: never OK at the lower, replaced result.
    assert fitted.status is Status.NOT_CONVERGED
    assert fitted.params is None
