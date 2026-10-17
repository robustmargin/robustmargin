"""``fit_dose_response`` on a control-only observation (a log axis whose only severity level is
the ``0.0`` control) must raise a clear ``ValueError``, not an unhelpful numpy internal error.

Without the guard, an **all-success** control-only observation raises
``ValueError: zero-size array to reduction operation maximum which has no identity`` on both
fitting paths (``GLM``, ``upper=1.0, lower=0.0``; and ``GenericLikelihoodModel``, ``upper=1.0,
lower="estimate"``), for every link (``probit``, ``logit``, ``cloglog``) -- there is only one
likelihood-contributing severity level, the control itself contributes none on a log axis,
and something downstream (span/centering-and-scale computation over an empty nonzero-severity
array) calls ``numpy.max``/``numpy.min`` on an array of length 0.

A control-only observation whose control *has failures* does not reach this at all: it is
``Status.CONTROL_INCOMPATIBLE`` before any fitting is attempted (``models.py``'s own pre-fit
check), which is why this file's fixture is deliberately all-success -- the case this guard
actually targets.
"""

from __future__ import annotations

import pytest

from robustmargin import Axis, Observations, fit_dose_response

_AXIS = Axis(name="severity", unit="unit", scale="log")

_LINKS: tuple[str, ...] = ("probit", "logit", "cloglog")
# (lower, id) -- upper is always 1.0 fixed; lower is 0.0 (GLM path) or "estimate" (generic path).
_LOWER_BY_PATH: tuple[tuple[float | str, str], ...] = (
    (0.0, "glm_path"),
    ("estimate", "generic_path"),
)


def _control_only_obs() -> Observations:
    return Observations.from_counts(
        _AXIS,
        severity=[0.0],
        successes=[50],
        trials=[50],
        outcome="success",
        direction="decreasing",
    )


class TestControlOnlyInputRaisesAClearValueError:
    @pytest.mark.parametrize("link", _LINKS)
    @pytest.mark.parametrize(
        "lower", [lower for lower, _ in _LOWER_BY_PATH], ids=[pid for _, pid in _LOWER_BY_PATH]
    )
    def test_raises_a_clear_value_error_naming_the_real_requirement(
        self, lower: float | str, link: str
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            fit_dose_response(
                _control_only_obs(), model="binomial", link=link, upper=1.0, lower=lower
            )

        message = str(excinfo.value)
        assert "zero-size array" not in message, (
            "the raised ValueError is still the unhelpful numpy internal one, not a message "
            f"this package wrote itself: {message!r}"
        )
        assert "at least one severity level besides the control" in message, (
            f"expected a message naming the real requirement, got {message!r}"
        )
