"""Coverage of every exact result ``censoring.classify`` returns on a failed fit, under a monotone
true curve.

``classify`` reads exact claims off one-sided exact tests at ``c4 = 1 - (1 - level) / 4`` (0.9875
at ``level = 0.95``). The error budget ``1 - level`` is split four ways, each part
``(1 - level) / 4``: the ascending run (``lo``), the top cell (``RIGHT``), the descending run
(``hi``) and the bottom cell (``hi = x_1``, the upper half of ``LEFT``, half of
``FAILS_AT_BASELINE``). Whatever kind the data selects, a returned claim is wrong only if one of
those four tests errs, so **for a fixed target the chance that any exact result is returned and
wrong is at most ``1 - level``**. Each run is a fixed-sequence test (Hsu & Berger, 1999, *JASA*
94:468-482): it can only err by passing (or failing) the *first* cell past the true threshold. A
rule that took each side's extreme over *every* cell, or tested each kind at ``level`` separately,
would not have this property.

One case is outside that statement, and the design that probes it is held to the same limit for a
different reason (see "What is asserted"): **exact ties.** If a tested cell's true rate equals the
target, the strict reading of a claim about that cell is covered only up to
``1.5 * (1 - level)``, not ``1 - level``. Design T is every cell at the target.
This is a simulation of the **method**, not the implementation: it draws counts from a known
monotone decreasing curve, runs the real ``classify`` on a failed fit, and counts how often the
returned result makes a wrong claim. Marked ``slow`` and excluded from the default run by
``pyproject.toml``'s ``-m "not slow"``; CI runs it in the ``coverage-simulations`` job. Run locally
with::

    pytest -m slow --no-cov tests/coverage/test_exact_bracket_coverage.py

**What counts as wrong** (the strict reading, so a cell whose true rate equals the target exactly
makes every claim about it wrong). The true success probabilities ``p`` are known per severity
(severity ``i`` has probability ``probabilities[i]``), and a result is wrong if any of its claims
fails against the target, ``0.9`` for every design:

* a ``lo`` (a bracket's or one-sided bound, or ``RIGHT``'s) claims the threshold exceeds ``lo``,
  so it is wrong unless ``p(lo) > target``;
* a ``hi`` (a bracket's or one-sided bound, or ``LEFT``'s) claims the threshold is below ``hi``,
  so it is wrong unless ``p(hi) < target``;
* ``LEFT`` also claims the control is above the target, so it is wrong unless ``p(0) > target``;
* ``FAILS_AT_BASELINE`` claims the control is below the target, so it is wrong unless
  ``p(0) < target``.

A result with no bounds makes no claim and is never wrong. For a design with no cell at the target
this is "the reported bound excludes the true threshold".

**The designs** (target ``0.9``, level ``0.95``; ``n`` is the trials per cell). Rates listed
in severity order starting at the control:

* **B** -- 4 levels just past the threshold, ``n = 200``; a bracket. The true threshold is between
  severities 3 and 4.
* **C** -- 8 levels just past the threshold, ``n = 1000``; a bracket. Less discreteness, so each
  false pass is nearer its nominal probability. Threshold between 3 and 4.
* **D** -- as C with the failing cell removed, so nothing confidently fails: a one-sided lower
  bound, and ``RIGHT`` wrongly in a few percent of replicates.
* **U** -- the mirror of C, to exercise the *descending* run (``hi``): six levels just *before* the
  threshold (rates 0.9001 and slightly above), then clearly failing levels. Threshold between 7
  and 8.
* **E** -- the ``RIGHT`` / ``LEFT`` conflict: a control at 1.0, one level at 0.9001 and one at
  0.8999, ``n = 1000``. ``LEFT`` is wrong if the 0.9001 level fails, ``RIGHT`` if the 0.8999 level
  passes; each is one test at ``c4``. Mostly no bounds, by construction.
* **T** (ties) -- every cell, the control included, exactly at the target: any exact claim is
  wrong under the strict reading.
* **K** (control at the target) -- the control exactly at the target, then levels just under it
  and clear failures.
* **R**, **L** -- power designs (below), with nothing near the target on one side.
**What is asserted.**

* For every design, the all-kinds wrong-claim rate is at most ``1 - level`` plus three standard
  errors of a proportion at that rate (``3 * sqrt(0.05 * 0.95 / 20000) = 0.0046``). The measured
  rates are about 0.5% (B), 0.9% (C), 1.5% (D), 1.0% (U), 1.9% (E), 3.4% for the ties design
  and 0.8% for K, so the margin is a statistical allowance, not a fitted
  one, and a different numpy build must still pass. It is tight enough to catch the failure it is
  for: a rule that tested each kind at ``level`` separately reaches 7.7% on D and 8.5% on E.
* **The limit does not mean the same thing on every design.** On B, C, D, U, E, R and L it is the
  proved bound plus a Monte Carlo margin, and exceeding it is a method finding. On T and K, which
  have cells exactly at the target (every cell on T, the control on K), the proved bound for exact
  ties is ``1.5 * (1 - level) = 0.075``, so the limit is only a regression guard on the measured
  3.4% (T) and 0.8% (K): exceeding it is a regression from that rate, not a refutation of anything
  proved.
* A bracket's **stressed side** (the side the design puts at risk) errs at most
  ``(1 - c4)`` of the time plus three standard errors at that rate
  (``0.0125 + 3 * sqrt(0.9875 * 0.0125 / 20000) = 0.0149``): each side of a bracket is a quarter
  share of the budget, so the joint limit above is too loose to catch a side computed at ``level``.
* **Not vacuous.** A design that is built to produce a kind of result must produce it in at least
  half the replicates (E, T and K are mostly no-bounds by construction and R and L are held to
  their power floors instead, so they are exempt), and a stress design must produce at least one
  wrong claim.
* **Power.** The split costs power, which these pin from below so that it cannot be bought
  silently: ``RIGHT`` fires in at least about 50% of replicates with eight levels at 0.95 and
  ``n = 200`` against a target of 0.9 (about 59% is measured), and ``LEFT`` in at least about 40%
  with eight levels at 0.85 (about 45%).
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Iterator

import numpy as np
import pytest

from robustmargin import Definition, Fit, Status, Threshold, censoring, threshold
from robustmargin.testing import fake_fit
from robustmargin.types import Cell, Censoring

_LEVEL = 0.95
_TARGET = 0.9
_REPLICATES = 20_000
_Z = 3.0
_ALPHA = 1.0 - _LEVEL
_MARGIN = _Z * math.sqrt(_ALPHA * _LEVEL / _REPLICATES)
_LIMIT = _ALPHA + _MARGIN

# Every exact test is at c4, and a stressed bracket side is one quarter share of the budget, so its
# own error limit is `1 - c4`, with the same three-standard-error margin taken at that rate.
_C4 = 1.0 - _ALPHA / 4.0
_SIDE_MARGIN = _Z * math.sqrt(_C4 * (1.0 - _C4) / _REPLICATES)
_SIDE_LIMIT = (1.0 - _C4) + _SIDE_MARGIN

# Power floors: "at least about" the measured rate, with a few points of room.
_RIGHT_POWER_FLOOR = 0.50
_LEFT_POWER_FLOOR = 0.40


# What the all-kinds limit means for a design, which decides how a failure is worded.
_PROVED = "proved"  # the limit is the proved bound plus a margin
_TIES = "ties"  # the proved bound is 1.5 * (1 - level); the limit guards the measured rate


@dataclasses.dataclass(frozen=True)
class _Design:
    name: str
    probabilities: tuple[float, ...]  # true success probability at severities 0, 1, 2, ...
    trials: int
    seed: int
    # The kind of result the design is built to produce; at least half the replicates must be it.
    # `None` for designs that are mostly no-bounds by construction.
    intended_kind: str | None
    # The bracket side the design puts at risk, "lo" or "hi"; only for bracket designs.
    stressed_side: str | None = None
    # Whether at least one wrong claim must occur (the design really stresses the method).
    must_stress: bool = True
    # What the all-kinds limit means for this design.
    basis: str = _PROVED


_NEAR_TARGET_C = tuple(0.8999 - 1e-5 * i for i in range(8))

_B = _Design(
    name="B: 4 levels just past the threshold, n=200",
    probabilities=(1.0, 0.999, 0.99, 0.95, 0.8999, 0.8998, 0.8997, 0.8996, 0.30),
    trials=200,
    seed=1,
    intended_kind="bracket",
    stressed_side="lo",
)
_C = _Design(
    name="C: 8 levels just past the threshold, n=1000",
    probabilities=(1.0, 0.999, 0.99, 0.95, *_NEAR_TARGET_C, 0.30),
    trials=1000,
    seed=3,
    intended_kind="bracket",
    stressed_side="lo",
)
_D = _Design(
    name="D: as C with nothing failing (one-sided lower), n=1000",
    probabilities=(1.0, 0.999, 0.99, 0.95, *_NEAR_TARGET_C),
    trials=1000,
    seed=4,
    intended_kind="one-sided lower",
)
# The mirror of C: six levels just *before* the threshold (monotone decreasing, all above the
# target, the last at 0.9001), then two clear failures.
_NEAR_TARGET_U = tuple(0.9001 + 1e-5 * (5 - i) for i in range(6))
_U = _Design(
    name="U: 6 levels just before the threshold (mirror of C), n=1000",
    probabilities=(1.0, 0.999, *_NEAR_TARGET_U, 0.5, 0.3),
    trials=1000,
    seed=5,
    intended_kind="bracket",
    stressed_side="hi",
)
# The RIGHT / LEFT conflict: the level below the threshold is a hair above the target and the one
# above it a hair below.
_E = _Design(
    name="E: RIGHT/LEFT conflict (control 1.0, then 0.9001 and 0.8999), n=1000",
    probabilities=(1.0, 0.9001, 0.8999),
    trials=1000,
    seed=6,
    intended_kind=None,
)
_T = _Design(
    name="T: every cell exactly at the target (ties), n=1000",
    probabilities=(_TARGET,) * 8,
    trials=1000,
    seed=7,
    intended_kind=None,
    basis=_TIES,
)
_K = _Design(
    name="K: the control exactly at the target, n=1000",
    probabilities=(_TARGET, 0.8999, 0.8998, 0.5, 0.3),
    trials=1000,
    seed=8,
    intended_kind=None,
    basis=_TIES,
)
# Power designs: eight levels at one rate, a control at 1.0.
_R = _Design(
    name="R: 8 levels at 0.95, n=200 (RIGHT power)",
    probabilities=(1.0, *(0.95,) * 8),
    trials=200,
    seed=9,
    intended_kind=None,
    must_stress=False,
)
_L = _Design(
    name="L: 8 levels at 0.85, n=200 (LEFT power)",
    probabilities=(1.0, *(0.85,) * 8),
    trials=200,
    seed=10,
    intended_kind=None,
    must_stress=False,
)

_DESIGNS = [_B, _C, _D, _U, _E, _T, _K, _R, _L]


@dataclasses.dataclass
class _Tally:
    results: int = 0
    wrong: int = 0
    lo_wrong: int = 0  # results whose `lo` claim is wrong
    hi_wrong: int = 0  # results whose `hi` claim is wrong


@pytest.fixture(scope="module", autouse=True)
def _memoised_exact_bound() -> Iterator[None]:
    """``exact_one_sided_bound`` is a pure function of ``(successes, trials, level, side)``, so
    memoising it changes no result, only the run time (it is a ``scipy.stats.binomtest`` call
    and ``classify`` makes a dozen per replicate)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            censoring,
            "exact_one_sided_bound",
            functools.lru_cache(maxsize=None)(censoring.exact_one_sided_bound),
        )
        yield


_Result = censoring.Classification | Threshold


def _claim_kind(result: _Result) -> str:
    if result.status is Status.FAILS_AT_BASELINE:
        return "FAILS_AT_BASELINE"
    if result.censoring is Censoring.RIGHT:
        return "RIGHT"
    if result.censoring is Censoring.LEFT:
        return "LEFT"
    if result.lo is None and result.hi is None:
        return "no bounds"
    if result.lo is not None and result.hi is not None:
        return "bracket"
    return "one-sided lower" if result.hi is None else "one-sided upper"


def _lo_is_wrong(result: _Result, p: tuple[float, ...], target: float) -> bool:
    """``lo`` (and ``RIGHT``'s bound) claims the threshold exceeds it: wrong unless the true rate
    there is above the target."""
    return result.lo is not None and not p[int(result.lo)] > target


def _hi_is_wrong(result: _Result, p: tuple[float, ...], target: float) -> bool:
    """``hi`` (and ``LEFT``'s bound) claims the threshold is below it: wrong unless the true rate
    there is below the target."""
    return result.hi is not None and not p[int(result.hi)] < target


def _other_claim_is_wrong(result: _Result, p: tuple[float, ...], target: float) -> bool:
    """The claims about the control: ``LEFT`` says it is above the target and
    ``FAILS_AT_BASELINE`` says it is below."""
    if result.censoring is Censoring.LEFT:
        return not p[0] > target
    if result.status is Status.FAILS_AT_BASELINE:
        return not p[0] < target
    return False


@functools.cache
def _simulate(design: _Design) -> dict[str, _Tally]:
    """Run the replicates of one design. Cached, so the tests that read it share one run; the
    tallies are not mutated after this returns."""
    base: Fit = fake_fit(status=Status.NOT_CONVERGED, monotonicity=None)
    probabilities = np.array(design.probabilities)
    trials = np.full(len(probabilities), design.trials)
    rng = np.random.default_rng(design.seed)
    tallies: dict[str, _Tally] = {}
    for _ in range(_REPLICATES):
        successes = rng.binomial(trials, probabilities)
        cells = tuple(
            Cell(severity=float(severity), successes=int(k), trials=int(n))
            for severity, (k, n) in enumerate(zip(successes, trials, strict=True))
        )
        result = censoring.classify(
            dataclasses.replace(base, cells=cells), target=_TARGET, level=_LEVEL
        )
        target = _TARGET
        tally = tallies.setdefault(_claim_kind(result), _Tally())
        lo_wrong = _lo_is_wrong(result, design.probabilities, target)
        hi_wrong = _hi_is_wrong(result, design.probabilities, target)
        other_wrong = _other_claim_is_wrong(result, design.probabilities, target)
        tally.results += 1
        tally.wrong += lo_wrong or hi_wrong or other_wrong
        tally.lo_wrong += lo_wrong
        tally.hi_wrong += hi_wrong
    return tallies


_KIND_ORDER = (
    "bracket",
    "one-sided lower",
    "one-sided upper",
    "RIGHT",
    "LEFT",
    "FAILS_AT_BASELINE",
    "no bounds",
)


def _what_exceeding_the_limit_means(design: _Design) -> str:
    """The tail of the failure message. The same limit is a proof-backed bound on some designs and
    only a guard on the measured rate on others, and the message must not claim more than that."""
    if design.basis == _TIES:
        return (
            f"the proved bound for exact ties is 1.5 * (1 - level) = {1.5 * _ALPHA:.3f}, so this "
            "design is not a proof failure; the limit above is a regression guard on the rate "
            "measured for this design (listed in the module docstring), and exceeding it is a "
            "regression from that rate"
        )
    return "this is a method finding, not a flaky test"


@pytest.mark.slow
@pytest.mark.parametrize("design", _DESIGNS, ids=lambda d: d.name.split(":")[0])
def test_every_exact_result_taken_together_is_wrong_at_most_one_minus_level_plus_margin(
    design: _Design,
) -> None:
    tallies = _simulate(design)
    all_wrong = sum(t.wrong for t in tallies.values()) / _REPLICATES

    print(
        f"\n{design.name} | level {_LEVEL}, target {_TARGET}, {_REPLICATES} replicates, "
        f"seed {design.seed}\n"
        f"  limit (1 - level + {_Z:g} SE) : {_LIMIT:.4f}  (margin {_MARGIN:.4f})"
    )
    for kind in _KIND_ORDER:
        if kind in tallies:
            t = tallies[kind]
            print(
                f"  {kind:17}: {t.results:6} results, {t.wrong:5} wrong "
                f"({t.wrong / _REPLICATES:.4f} of all replicates)"
            )
    print(f"  all kinds wrong   : {all_wrong:.4f}")

    # Not vacuous: the design produces the kind of result it is meant to in most replicates, so a
    # rate under the limit is not just "almost nothing was reported".
    if design.intended_kind is not None:
        intended = tallies.get(design.intended_kind, _Tally()).results
        assert intended >= _REPLICATES // 2, (
            f"only {intended} of {_REPLICATES} replicates gave a {design.intended_kind!r} result "
            f"on design {design.name}; the design no longer exercises the branch it is for"
        )
    if design.must_stress:
        assert all_wrong > 0.0, (
            f"no replicate made a wrong claim on design {design.name}; the design no longer "
            "stresses the method"
        )
    assert all_wrong <= _LIMIT, (
        f"all-kinds wrong-claim rate {all_wrong:.4f} exceeds {_ALPHA:.2f} + {_MARGIN:.4f} on "
        f"design {design.name}; {_what_exceeding_the_limit_means(design)}"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "design", [d for d in _DESIGNS if d.stressed_side is not None], ids=lambda d: d.name[0]
)
def test_stressed_bracket_side_error_is_at_most_one_minus_c4_plus_margin(
    design: _Design,
) -> None:
    """Each side of a bracket is a quarter share of the budget, tested at ``c4``, so the side the
    design puts at risk must err at most ``1 - c4`` of the time (plus a Monte Carlo margin), not
    merely stay inside the joint ``1 - level``. A bracket side computed at ``level`` errs about
    4.3% on C and passes the joint limit; it does not pass this one."""
    tallies = _simulate(design)
    bracket = tallies["bracket"]
    side_wrong = bracket.lo_wrong if design.stressed_side == "lo" else bracket.hi_wrong
    other_wrong = bracket.hi_wrong if design.stressed_side == "lo" else bracket.lo_wrong
    side_rate = side_wrong / _REPLICATES

    print(
        f"\n{design.name} | stressed side {design.stressed_side!r}, "
        f"limit {_SIDE_LIMIT:.4f} (1 - c4 = {1.0 - _C4:.4f}, margin {_SIDE_MARGIN:.4f})"
        f"\n  {design.stressed_side} wrong in {side_wrong} of {bracket.results} brackets "
        f"({side_rate:.4f} of all replicates); the other side wrong in {other_wrong}"
    )

    # Not vacuous: the design stresses the side it names. A tally of zero would pass any limit.
    assert bracket.results >= _REPLICATES // 2
    assert side_wrong > 0, (
        f"the {design.stressed_side!r} side was never wrong on design {design.name}; "
        "the design no longer stresses it"
    )
    assert side_rate <= _SIDE_LIMIT, (
        f"bracket {design.stressed_side!r}-side wrong-claim rate {side_rate:.4f} exceeds "
        f"{1.0 - _C4:.4f} + {_SIDE_MARGIN:.4f} on design {design.name}; each bracket side is a "
        "quarter share of the budget, tested at c4 = 1 - (1 - level) / 4"
    )


@pytest.mark.slow
def test_right_fires_in_about_half_the_replicates_with_eight_levels_at_0_95() -> None:
    """Power floor. The top cell passes at ``c4`` only when its lower bound at 0.9875 clears 0.9:
    with ``n = 200`` and a true rate of 0.95 that is about 59% of the time."""
    tallies = _simulate(_R)
    rate = tallies.get("RIGHT", _Tally()).results / _REPLICATES

    print(f"\n{_R.name}: RIGHT in {rate:.4f} of replicates (floor {_RIGHT_POWER_FLOOR:.2f})")

    assert rate >= _RIGHT_POWER_FLOOR, (
        f"RIGHT fires in only {rate:.4f} of replicates on {_R.name}; the split of the error "
        "budget has cost more power than recorded"
    )


@pytest.mark.slow
def test_left_fires_in_about_forty_percent_of_replicates_with_eight_levels_at_0_85() -> None:
    """Power floor. The lowest nonzero cell fails at ``c4`` only when its upper bound at 0.9875
    is below 0.9, and the control (1.0) passes: with ``n = 200`` and a true rate of 0.85 that is
    about 45% of the time."""
    tallies = _simulate(_L)
    rate = tallies.get("LEFT", _Tally()).results / _REPLICATES

    print(f"\n{_L.name}: LEFT in {rate:.4f} of replicates (floor {_LEFT_POWER_FLOOR:.2f})")

    assert rate >= _LEFT_POWER_FLOOR, (
        f"LEFT fires in only {rate:.4f} of replicates on {_L.name}; the split of the error "
        "budget has cost more power than recorded"
    )


# ---------------------------------------------------------------------------------------------
# The increasing orientation of design C.
#
# Events are drawn from ``1 - p`` for design C's success probabilities ``p`` and solved at the
# events-scale target ``1 - 0.9``, so the true curve crosses it between severities 3 and 4 exactly
# as C's does, with the claims reading the other way: a ``lo`` says the rate is still *below* the
# target there, and so on. A claim is judged by the same rule as on the decreasing curve once the
# probabilities and the target are mirrored, which is why ``_lo_is_wrong`` and its siblings are
# reused with ``_C.probabilities`` and ``_TARGET``. The seed is independent of C's.
# ---------------------------------------------------------------------------------------------

_SEED_INCREASING_C = 13


@pytest.mark.slow
def test_increasing_design_c_exact_claims_are_wrong_at_most_one_minus_level_plus_margin() -> None:
    design = _C
    base: Fit = fake_fit(status=Status.NOT_CONVERGED, direction="increasing", monotonicity=None)
    event_probabilities = 1.0 - np.array(design.probabilities)
    trials = np.full(len(event_probabilities), design.trials)
    rng = np.random.default_rng(_SEED_INCREASING_C)
    events_target = 1.0 - _TARGET

    wrong = brackets = 0
    for _ in range(_REPLICATES):
        events = rng.binomial(trials, event_probabilities)
        cells = tuple(
            Cell(severity=float(severity), successes=int(k), trials=int(n))
            for severity, (k, n) in enumerate(zip(events, trials, strict=True))
        )
        result = threshold(
            dataclasses.replace(base, cells=cells),
            definition=Definition.absolute(events_target),
            interval_method="profile",
            level=_LEVEL,
        )
        brackets += _claim_kind(result) == "bracket"
        wrong += (
            _lo_is_wrong(result, design.probabilities, _TARGET)
            or _hi_is_wrong(result, design.probabilities, _TARGET)
            or _other_claim_is_wrong(result, design.probabilities, _TARGET)
        )

    rate = wrong / _REPLICATES
    print(
        f"\nincreasing orientation of design C | level {_LEVEL}, events-scale target "
        f"{events_target:.4f}, {_REPLICATES} replicates, seed {_SEED_INCREASING_C}\n"
        f"  limit (1 - level + {_Z:g} SE) : {_LIMIT:.4f}\n"
        f"  brackets          : {brackets}\n"
        f"  all kinds wrong   : {rate:.4f}"
    )
    assert brackets >= _REPLICATES // 2, "the design no longer produces brackets"
    assert rate > 0.0, "no replicate made a wrong claim; the design no longer stresses the method"
    assert rate <= _LIMIT, (
        f"all-kinds wrong-claim rate {rate:.4f} exceeds {_ALPHA:.2f} + {_MARGIN:.4f} on the "
        "increasing orientation of design C; this is a method finding, not a flaky test"
    )


# ---------------------------------------------------------------------------------------------
# An increasing outcome, ``baseline_multiple(k)`` on a failed fit: a refusal, not a coverage claim.
#
# On a failed fit the only baseline is the control's observed rate, an estimate. Exact claims read
# against ``k`` times that estimate are about a target the caller did not ask for, and judged
# against ``k * P(0)`` they err far above ``1 - level`` in simulation, so the definition is refused
# on a failed fit. This draws counts from a design with a small control rate and levels around the
# plug-in target and checks that every replicate raises, whatever the control count (zero events
# included) and whatever the plug-in works out to. Nothing is judged about the rate of wrong claims.
# ---------------------------------------------------------------------------------------------

_SEED_INCREASING_MULTIPLE = 14
_MULTIPLE_EVENT_PROBABILITIES = (0.04, 0.06, 0.10, 0.20, 0.40)


@pytest.mark.slow
def test_increasing_baseline_multiple_on_a_failed_fit_always_raises() -> None:
    base: Fit = fake_fit(status=Status.NOT_CONVERGED, direction="increasing", monotonicity=None)
    event_probabilities = np.array(_MULTIPLE_EVENT_PROBABILITIES)
    trials = np.full(len(event_probabilities), 500)
    trials[0] = 100
    rng = np.random.default_rng(_SEED_INCREASING_MULTIPLE)

    zero_controls = 0
    for _ in range(2000):
        events = rng.binomial(trials, event_probabilities)
        zero_controls += events[0] == 0
        cells = tuple(
            Cell(severity=float(severity), successes=int(k), trials=int(n))
            for severity, (k, n) in enumerate(zip(events, trials, strict=True))
        )
        with pytest.raises(ValueError, match="NOT_CONVERGED"):
            threshold(
                dataclasses.replace(base, cells=cells),
                definition=Definition.baseline_multiple(2.0),
                interval_method="profile",
                level=_LEVEL,
            )

    assert zero_controls > 0, "the design never drew a control with no events"


# ---------------------------------------------------------------------------------------------
# ``baseline_fraction(p)`` on a failed fit: a refusal, not a coverage claim.
#
# On a failed fit the only baseline is the control's observed rate, an estimate. Exact claims read
# against ``p`` times that estimate are about a target the caller did not ask for, and the target
# moves with every replicate's control count, so the definition is refused on a failed fit. This
# draws counts from a design whose control sits near the plug-in target and checks that every
# replicate raises, whatever the control count. Nothing is judged about the rate of wrong claims.
# ---------------------------------------------------------------------------------------------

_SEED_BASELINE_FRACTION = 15
_FRACTION_PROBABILITIES = (0.9565, 0.92, 0.88, 0.80)


@pytest.mark.slow
def test_baseline_fraction_on_a_failed_fit_always_raises() -> None:
    base: Fit = fake_fit(status=Status.NOT_CONVERGED, monotonicity=None)
    probabilities = np.array(_FRACTION_PROBABILITIES)
    trials = np.full(len(probabilities), 500)
    trials[0] = 100
    rng = np.random.default_rng(_SEED_BASELINE_FRACTION)

    distinct_controls: set[int] = set()
    for _ in range(2000):
        successes = rng.binomial(trials, probabilities)
        distinct_controls.add(int(successes[0]))
        cells = tuple(
            Cell(severity=float(severity), successes=int(k), trials=int(n))
            for severity, (k, n) in enumerate(zip(successes, trials, strict=True))
        )
        with pytest.raises(ValueError, match="NOT_CONVERGED"):
            threshold(
                dataclasses.replace(base, cells=cells),
                definition=Definition.baseline_fraction(0.957),
                interval_method="profile",
                level=_LEVEL,
            )

    assert len(distinct_controls) >= 5, "the design drew too few distinct control counts"
