"""Coverage of the failed-fit exact bracket under a monotone true curve.

``censoring.classify`` on a ``SEPARATION`` / ``NOT_CONVERGED`` fit reports an exact bracket (or
a one-sided exact bound) taken from the tested cells. Its coverage claim: each side at
``(1 + level) / 2`` so the pair covers at least ``level``; a one-sided bound at ``level``. A
rule that took each side's extreme over *every* cell would break that: with ``k`` cells near the
target the error of each side is up to about ``k * (1 - c)``. The fixed-sequence scans the
implementation uses (Hsu & Berger, 1999, *JASA* 94:468-482) err on each side only if the *first*
cell past the true threshold errs.

This is a simulation of the **method**, not the implementation: it draws counts from a known
monotone decreasing curve, runs the real ``classify``, and counts how often the reported bound
excludes the true threshold. A result "makes a wrong claim" when its ``lo`` is at or above the
first tested severity past the threshold, or its ``hi`` is at or below the last one before it. A
result with no bounds makes no claim and is never counted wrong.

Marked ``slow`` and excluded from the default run by ``pyproject.toml``'s ``-m "not slow"``; CI
runs it in the ``coverage-simulations`` job. Run locally with::

    pytest -m slow --no-cov tests/coverage/test_exact_bracket_coverage.py

**The designs** (target ``0.9``, level ``0.95``):

* **B** -- 4 tested levels just past the threshold, ``n = 200`` per cell, a bracket. The true
  threshold is between severities 3 and 4.
* **C** -- 8 levels just past the threshold, ``n = 1000``, a bracket. Less discreteness, so each
  false pass is nearer its nominal probability. Threshold between 3 and 4.
* **D** -- as C with the failing cell removed, so nothing confidently fails and the result is a
  one-sided lower bound, claimed at ``level`` itself. Threshold between 3 and 4.
* **U** -- the mirror image of C, to exercise the *descending* scan (``hi``). Six levels sit just
  *before* the threshold (true rate 0.9001 and slightly above), with clearly failing levels
  after them, so the only way for ``hi`` to be wrong is for the last level before the threshold
  to be wrongly called a failure. B, C and D have no level below the threshold near the target,
  so on them a wrong ``hi`` cannot occur and the upper scan is not tested by simulation at all.
  The threshold is between severities 7 and 8.

(A steep curve with one level near the target on each side is trivially 0% under both rules
and is not repeated here.) Each design's seed is fixed in the code, so every reported rate
reproduces exactly.

**U gives a bracket, not a one-sided upper bound.** The lowest nonzero level has a true rate of
0.999, so at ``n = 1000`` its lower bound (about 0.99) clears the target in every replicate and
the lower scan is never empty; a one-sided upper result needs the lower side to be undetermined.
That follows from "control 1.0, 0.999, then the near-target levels", and it is what leaves the
bracket's ``hi`` as the only side at risk. The one-sided-upper rescan at ``level`` is covered by
``tests/unit/test_censoring_fixed_sequence.py``, not by a simulation. For the same reason U
never reaches the control in the descending scan: the control passes, so the scan has already
stopped before it.

**What is asserted.** The failed-fit fallback is held to the coverage claim above: the
fraction of replicates in which the reported *exact-bracket* bound excludes the true threshold is
at most ``1 - level`` plus a Monte Carlo margin. ``RIGHT`` and ``LEFT`` are different claims --
each rests on one cell chosen in advance and has its own guarantee of ``1 - level`` -- so they
are held to the same bound separately, not folded into the fallback's. On B and C neither can
fire (the top cell fails, the control passes), so the all-claims rate is asserted too. On D the
top cell sits just under the target, so ``RIGHT`` fires, wrongly, in about 3% of replicates, and
the all-claims rate (about 7.7%) is **recorded, not asserted**: it is the sum of two separately
valid ``1 - level`` guarantees and is a finding to report, not something a fixed-sequence
scan can change.

**The margin.** ``_Z = 3`` standard errors of a proportion at the nominal rate:
``3 * sqrt(0.05 * 0.95 / 20000) = 0.0046``. Were the true rate exactly 5%, a replicate count
above the limit would have probability about 0.13% (one-sided); the rates measured under the
fixed-sequence rule are about 1.2% (B), 2.0% (C) and 4.3% (D, the closest to the claim, 7 standard
errors under the limit), so the real false-alarm probability is far smaller. The margin is
statistical, not fitted to the seeds: a different numpy build draws a different stream, and the
assertion must still hold. It is also tight enough to catch a max-anywhere rule -- 6.0% on B
(3.6 standard errors over the limit) and 13.6% on C and 28% of D's one-sided results, far over.

**The per-side limit.** The joint limit above is the claim made for the *pair*, and it
is loose against a regression that computes each bracket side at ``level`` instead of
``(1 + level) / 2``: that gives about 4.3% on C, which still passes 5.46%. So the bracket's
**stressed side** -- the side the design puts at risk -- is also held to its own claim,
``1 - c`` with ``c = (1 + level) / 2``, plus three standard errors of a proportion at that rate:
``0.025 + 3 * sqrt(0.975 * 0.025 / 20000) = 0.0283``. The stressed side is ``lo`` on B and C and
``hi`` on U. D has no stressed bracket side: its lower bound is one-sided, claimed at ``level``,
and is held by the fallback limit above.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Iterator

import numpy as np
import pytest

from robustmargin import Fit, Status, censoring
from robustmargin.testing import fake_fit
from robustmargin.types import Cell, Censoring

_LEVEL = 0.95
_TARGET = 0.9
_REPLICATES = 20_000
_Z = 3.0
_MARGIN = _Z * math.sqrt((1.0 - _LEVEL) * _LEVEL / _REPLICATES)
_LIMIT = (1.0 - _LEVEL) + _MARGIN

# A bracket side is claimed at the Bonferroni split `c`, so its own error limit is `1 - c`, with
# the same three-standard-error Monte Carlo margin taken at that rate.
_SIDE_CONFIDENCE = (1.0 + _LEVEL) / 2.0
_SIDE_MARGIN = _Z * math.sqrt(_SIDE_CONFIDENCE * (1.0 - _SIDE_CONFIDENCE) / _REPLICATES)
_SIDE_LIMIT = (1.0 - _SIDE_CONFIDENCE) + _SIDE_MARGIN


@dataclasses.dataclass(frozen=True)
class _Design:
    name: str
    probabilities: tuple[float, ...]  # true success probability at severities 0, 1, 2, ...
    trials: int
    seed: int
    all_claims_asserted: bool
    # The kind of result the design is built to produce; at least half the replicates must be it.
    intended_kind: str
    # The true threshold lies strictly between these two tested severities. A claim is wrong when
    # its `lo` is at or above `first_past` or its `hi` is at or below `last_before`.
    last_before: float
    first_past: float
    # The bracket side the design puts at risk, "lo" or "hi"; used only for bracket designs.
    stressed_side: str


_NEAR_TARGET_C = tuple(0.8999 - 1e-5 * i for i in range(8))

_B = _Design(
    name="B: 4 levels just past the threshold, n=200",
    probabilities=(1.0, 0.999, 0.99, 0.95, 0.8999, 0.8998, 0.8997, 0.8996, 0.30),
    trials=200,
    seed=1,
    all_claims_asserted=True,
    intended_kind="bracket",
    last_before=3.0,
    first_past=4.0,
    stressed_side="lo",
)
_C = _Design(
    name="C: 8 levels just past the threshold, n=1000",
    probabilities=(1.0, 0.999, 0.99, 0.95, *_NEAR_TARGET_C, 0.30),
    trials=1000,
    seed=3,
    all_claims_asserted=True,
    intended_kind="bracket",
    last_before=3.0,
    first_past=4.0,
    stressed_side="lo",
)
_D = _Design(
    name="D: as C with nothing failing (one-sided lower), n=1000",
    probabilities=(1.0, 0.999, 0.99, 0.95, *_NEAR_TARGET_C),
    trials=1000,
    seed=4,
    all_claims_asserted=False,
    intended_kind="one-sided lower",
    last_before=3.0,
    first_past=4.0,
    stressed_side="lo",
)
# The mirror of C: six levels just *before* the threshold (monotone decreasing, all above the
# target, the last at 0.9001), then two clear failures. The wrong claim is a `hi` at or below the
# last level before the threshold, severity 7.
_NEAR_TARGET_U = tuple(0.9001 + 1e-5 * (5 - i) for i in range(6))
_U = _Design(
    name="U: 6 levels just before the threshold (mirror of C), n=1000",
    probabilities=(1.0, 0.999, *_NEAR_TARGET_U, 0.5, 0.3),
    trials=1000,
    seed=5,
    all_claims_asserted=True,
    intended_kind="bracket",
    last_before=7.0,
    first_past=8.0,
    stressed_side="hi",
)


@dataclasses.dataclass
class _Tally:
    results: int = 0
    wrong: int = 0
    lo_wrong: int = 0  # results whose `lo` excludes the true threshold
    hi_wrong: int = 0  # results whose `hi` excludes the true threshold


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


def _claim_kind(result: censoring.Classification) -> str:
    if result.censoring is Censoring.RIGHT:
        return "RIGHT"
    if result.censoring is Censoring.LEFT:
        return "LEFT"
    if result.lo is None and result.hi is None:
        return "no bounds"
    if result.lo is not None and result.hi is not None:
        return "bracket"
    return "one-sided lower" if result.hi is None else "one-sided upper"


def _lo_is_wrong(result: censoring.Classification, design: _Design) -> bool:
    return result.lo is not None and result.lo >= design.first_past


def _hi_is_wrong(result: censoring.Classification, design: _Design) -> bool:
    return result.hi is not None and result.hi <= design.last_before


@functools.cache
def _simulate(design: _Design) -> dict[str, _Tally]:
    """Run the replicates of one design. Cached, so the two tests that read it share one run;
    the tallies are not mutated after this returns."""
    base: Fit = fake_fit(status=Status.NOT_CONVERGED, monotonicity=None)
    probabilities = np.array(design.probabilities)
    rng = np.random.default_rng(design.seed)
    tallies: dict[str, _Tally] = {}
    for _ in range(_REPLICATES):
        successes = rng.binomial(design.trials, probabilities)
        cells = tuple(
            Cell(severity=float(severity), successes=int(k), trials=design.trials)
            for severity, k in enumerate(successes)
        )
        result = censoring.classify(
            dataclasses.replace(base, cells=cells), target=_TARGET, level=_LEVEL
        )
        tally = tallies.setdefault(_claim_kind(result), _Tally())
        lo_wrong = _lo_is_wrong(result, design)
        hi_wrong = _hi_is_wrong(result, design)
        tally.results += 1
        tally.wrong += lo_wrong or hi_wrong
        tally.lo_wrong += lo_wrong
        tally.hi_wrong += hi_wrong
    return tallies


def _wrong(tallies: dict[str, _Tally], *kinds: str) -> int:
    return sum(tallies[k].wrong for k in kinds if k in tallies)


_FALLBACK_KINDS = ("bracket", "one-sided lower", "one-sided upper")
_DESIGNS = [_B, _C, _D, _U]


@pytest.mark.slow
@pytest.mark.parametrize("design", _DESIGNS, ids=lambda d: d.name.split(":")[0])
def test_exact_bracket_wrong_claim_rate_is_at_most_one_minus_level_plus_monte_carlo_margin(
    design: _Design,
) -> None:
    tallies = _simulate(design)

    fallback_wrong = _wrong(tallies, *_FALLBACK_KINDS) / _REPLICATES
    right_wrong = _wrong(tallies, "RIGHT") / _REPLICATES
    left_wrong = _wrong(tallies, "LEFT") / _REPLICATES
    all_wrong = sum(t.wrong for t in tallies.values()) / _REPLICATES

    print(
        f"\n{design.name} | level {_LEVEL}, target {_TARGET}, {_REPLICATES} replicates, "
        f"seed {design.seed}\n"
        f"  limit (1 - level + {_Z:g} SE) : {_LIMIT:.4f}  (margin {_MARGIN:.4f})"
    )
    for kind in ("bracket", "one-sided lower", "one-sided upper", "RIGHT", "LEFT", "no bounds"):
        if kind in tallies:
            t = tallies[kind]
            print(
                f"  {kind:16}: {t.results:6} results, {t.wrong:5} wrong "
                f"({t.wrong / _REPLICATES:.4f} of all replicates)"
            )
    print(
        f"  exact-bound fallback wrong : {fallback_wrong:.4f}\n"
        f"  RIGHT wrong / LEFT wrong   : {right_wrong:.4f} / {left_wrong:.4f}\n"
        f"  all claims wrong           : {all_wrong:.4f}"
        + ("" if design.all_claims_asserted else "  (recorded, not asserted)")
    )

    # Not vacuous: the design produces the kind of result it is meant to in most replicates, so a
    # rate under the limit is not just "almost nothing was reported".
    intended = tallies.get(design.intended_kind, _Tally()).results
    assert intended >= _REPLICATES // 2, (
        f"only {intended} of {_REPLICATES} replicates gave a {design.intended_kind!r} result on "
        f"design {design.name}; the design no longer exercises the branch it is for"
    )
    assert fallback_wrong <= _LIMIT, (
        f"exact-bound fallback wrong-claim rate {fallback_wrong:.4f} exceeds "
        f"{1.0 - _LEVEL:.2f} + {_MARGIN:.4f} on design {design.name}; this is a method finding, "
        "not a flaky test"
    )
    assert right_wrong <= _LIMIT
    assert left_wrong <= _LIMIT
    if design.all_claims_asserted:
        assert all_wrong <= _LIMIT, (
            f"all-claims wrong rate {all_wrong:.4f} exceeds {_LIMIT:.4f} on {design.name}"
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "design", [d for d in _DESIGNS if d.intended_kind == "bracket"], ids=lambda d: d.name[0]
)
def test_stressed_bracket_side_error_is_at_most_one_minus_side_confidence_plus_margin(
    design: _Design,
) -> None:
    """Each side of a bracket is claimed at ``c = (1 + level) / 2``, so the side the design puts
    at risk must err at most ``1 - c`` of the time, not merely stay inside the joint
    ``1 - level``. A bracket computed at ``level`` per side errs about 4.3% on C and passes the
    joint limit; it does not pass this one."""
    tallies = _simulate(design)
    bracket = tallies["bracket"]
    side_wrong = bracket.lo_wrong if design.stressed_side == "lo" else bracket.hi_wrong
    other_wrong = bracket.hi_wrong if design.stressed_side == "lo" else bracket.lo_wrong
    side_rate = side_wrong / _REPLICATES

    print(
        f"\n{design.name} | stressed side {design.stressed_side!r}, "
        f"limit {_SIDE_LIMIT:.4f} (1 - c = {1.0 - _SIDE_CONFIDENCE:.3f}, margin {_SIDE_MARGIN:.4f})"
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
        f"{1.0 - _SIDE_CONFIDENCE:.3f} + {_SIDE_MARGIN:.4f} on design {design.name}; each "
        "bracket side must be claimed at (1 + level) / 2"
    )
