"""The exact cluster sign test behind ``check_monotonicity(obs, dependence="clustered")``.

For each adjacent pair of severity levels, a cluster contributes only when it is observed at both
levels with equal trials at both (so the comparison of its two rates is a comparison of counts and
the test stays exact). Its sign is "reversed" when its rate moves against ``obs.direction``;
equal rates are ties and are dropped. ``p = P(Binom(reversed + ordered, 1/2) >= reversed)``, one
sided, Holm-adjusted across all adjacent pairs.

Oracles are ``scipy.stats.binomtest`` and a hand-written Holm step-down. The warning wording is not
fixed by the specification, so warnings are matched on stated keyword alternatives
(``_UNBALANCED_WORDS`` and friends); tighten them to the implemented text.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest
from _cluster_support import LOG, Row, counts_obs, simulate_rows
from scipy.stats import binomtest

from robustmargin import ClusterAdjacentPair, MonotonicityCheck, Observations, check_monotonicity

_UNBALANCED_WORDS = ("unbalanced", "not balanced", "balanced")
_POWERLESS_WORDS = ("power", "cannot flag", "cannot be flagged", "minimum")
_NO_SHARED_WORDS = ("no cluster", "shared", "both levels")


def _holm(p_values: Sequence[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _hand_rows(
    levels: Sequence[float], per_cluster: Sequence[Sequence[int]], trials: int = 10
) -> list[Row]:
    rows: list[Row] = []
    for c, successes in enumerate(per_cluster):
        for level, k in zip(levels, successes, strict=True):
            rows.append((level, f"k{c:02d}", k, trials))
    return rows


def _check(
    rows: list[Row], *, direction: str = "decreasing", alpha: float = 0.05
) -> MonotonicityCheck:
    return check_monotonicity(
        counts_obs(rows, direction=direction), dependence="clustered", alpha=alpha
    )


def _pairs(check: MonotonicityCheck) -> tuple[ClusterAdjacentPair, ...]:
    pairs = check.pairs
    assert all(isinstance(p, ClusterAdjacentPair) for p in pairs)
    return cast("tuple[ClusterAdjacentPair, ...]", tuple(pairs))


def _contains(warnings: Sequence[str], words: Sequence[str]) -> bool:
    return any(any(w in text.lower() for w in words) for text in warnings)


# Three levels, twelve clusters. Pair (1, 2): 9 ordered, 2 reversed, 1 tie. Pair (2, 3): 8
# reversed, 1 ordered, 3 ties.
def _schedule_rows() -> list[Row]:
    per_cluster = [
        [9, 5, 6],  # ordered, reversed
        [9, 5, 6],
        [9, 5, 6],
        [9, 5, 6],
        [9, 5, 6],
        [9, 5, 6],
        [9, 5, 6],
        [9, 5, 6],  # 8 clusters: ordered in pair 1, reversed in pair 2
        [9, 6, 5],  # ordered, ordered
        [5, 9, 9],  # reversed, tie
        [5, 9, 9],  # reversed, tie
        [7, 7, 7],  # tie, tie
    ]
    return _hand_rows((1.0, 2.0, 3.0), per_cluster)


def test_cluster_sign_test_reports_reversed_ordered_and_tied_cluster_counts() -> None:
    check = _check(_schedule_rows())

    first, second = _pairs(check)
    assert (first.clusters_reversed, first.clusters_ordered, first.clusters_tied) == (2, 9, 1)
    assert (second.clusters_reversed, second.clusters_ordered, second.clusters_tied) == (8, 1, 3)


def test_cluster_sign_test_p_values_match_scipy_binomtest_with_ties_dropped() -> None:
    check = _check(_schedule_rows())

    expected = [
        float(binomtest(2, 11, 0.5, alternative="greater").pvalue),
        float(binomtest(8, 9, 0.5, alternative="greater").pvalue),
    ]
    assert [p.p_value for p in _pairs(check)] == pytest.approx(expected, rel=1e-12)


def test_cluster_sign_test_holm_adjusts_across_all_adjacent_pairs_and_flags_below_alpha() -> None:
    check = _check(_schedule_rows())

    expected = _holm([p.p_value for p in _pairs(check)])
    assert [p.holm_adjusted_p for p in _pairs(check)] == pytest.approx(expected, rel=1e-12)
    assert [p.flagged for p in _pairs(check)] == [a < 0.05 for a in expected]
    assert check.any_flagged is True
    assert check.method == "cluster_sign_test_one_sided_holm"
    assert check.dependence == "clustered"


def test_cluster_sign_test_with_no_reversals_has_p_value_one_for_zero_or_all_ordered() -> None:
    rows = _hand_rows((1.0, 2.0), [[9, 5]] * 6)

    (pair,) = _pairs(_check(rows))

    assert pair.clusters_reversed == 0 and pair.clusters_ordered == 6
    assert pair.p_value == pytest.approx(1.0)
    assert pair.flagged is False


def test_cluster_sign_test_p_value_is_one_when_every_shared_cluster_is_tied() -> None:
    rows = _hand_rows((1.0, 2.0), [[6, 6]] * 5)

    (pair,) = _pairs(_check(rows))

    assert (pair.clusters_reversed, pair.clusters_ordered, pair.clusters_tied) == (0, 0, 5)
    assert pair.p_value == 1.0


def test_cluster_sign_test_excludes_unbalanced_clusters_and_counts_them() -> None:
    balanced = _hand_rows((1.0, 2.0), [[9, 5]] * 3 + [[5, 9]] * 7)
    unbalanced: list[Row] = []
    for c in range(3):  # trials differ between the levels: excluded, whatever their rates
        unbalanced += [(1.0, f"u{c}", 5, 10), (2.0, f"u{c}", 18, 20)]

    clean = _check(balanced)
    mixed = _check(balanced + unbalanced)

    (clean_pair,) = _pairs(clean)
    (pair,) = _pairs(mixed)
    assert pair.clusters_unbalanced == 3
    assert clean_pair.clusters_unbalanced == 0
    assert (pair.clusters_reversed, pair.clusters_ordered, pair.clusters_tied) == (
        clean_pair.clusters_reversed,
        clean_pair.clusters_ordered,
        clean_pair.clusters_tied,
    )
    assert pair.p_value == clean_pair.p_value
    assert _contains(mixed.warnings, _UNBALANCED_WORDS)
    assert not _contains(clean.warnings, _UNBALANCED_WORDS)


def test_cluster_sign_test_does_not_count_a_cluster_observed_at_only_one_level() -> None:
    shared = _hand_rows((1.0, 2.0), [[9, 5]] * 6 + [[5, 9]] * 4)
    lone = [(1.0, "only_low", 0, 10), (2.0, "only_high", 10, 10)]

    (pair,) = _pairs(_check(shared + lone))

    assert (pair.clusters_reversed, pair.clusters_ordered, pair.clusters_tied) == (4, 6, 0)
    assert pair.clusters_unbalanced == 0


def test_cluster_sign_test_for_an_increasing_outcome_counts_a_falling_rate_as_reversed() -> None:
    rows = _schedule_rows()
    decreasing = _check(rows)
    flipped = [(sev, name, n - k, n) for sev, name, k, n in rows]

    increasing = _check(flipped, direction="increasing")

    for a, b in zip(_pairs(decreasing), _pairs(increasing), strict=True):
        assert (a.clusters_reversed, a.clusters_ordered, a.clusters_tied) == (
            b.clusters_reversed,
            b.clusters_ordered,
            b.clusters_tied,
        )
        assert a.p_value == pytest.approx(b.p_value, rel=1e-12)


def test_cluster_sign_test_is_invariant_to_cluster_labels_and_row_order() -> None:
    rows = _schedule_rows()
    relabelled = [(sev, "q" + name[::-1], k, n) for sev, name, k, n in rows]
    shuffled = [relabelled[i] for i in np.random.default_rng(5).permutation(len(relabelled))]

    a = _check(rows)
    b = _check(shuffled)

    assert [p.p_value for p in _pairs(a)] == [p.p_value for p in _pairs(b)]
    assert [p.holm_adjusted_p for p in _pairs(a)] == [p.holm_adjusted_p for p in _pairs(b)]


def test_cluster_sign_test_ignores_how_many_trials_a_balanced_cluster_has_unweighted() -> None:
    """Per-cluster differences are not weighted: a huge cluster counts once."""
    small = _hand_rows((1.0, 2.0), [[9, 5]] * 4 + [[5, 9]] * 4)
    big = [(1.0, "big", 500, 1000), (2.0, "big", 900, 1000)]  # one reversed cluster, 100x trials

    (pair,) = _pairs(_check(small + big))

    assert (pair.clusters_reversed, pair.clusters_ordered) == (5, 4)


def test_cluster_sign_test_warns_when_it_cannot_flag_anything() -> None:
    rows = _hand_rows((1.0, 2.0), [[5, 9]] * 3)  # best possible p with 3 clusters is 1/8

    check = _check(rows)

    assert _pairs(check)[0].flagged is False
    assert _contains(check.warnings, _POWERLESS_WORDS)


def test_cluster_sign_test_does_not_warn_of_powerlessness_with_enough_clusters() -> None:
    check = _check(_hand_rows((1.0, 2.0), [[5, 9]] * 8 + [[9, 5]] * 4))

    assert not _contains(check.warnings, _POWERLESS_WORDS)
    assert not _contains(check.warnings, _UNBALANCED_WORDS)


def test_cluster_sign_test_warns_when_no_cluster_is_observed_at_both_levels() -> None:
    rows = [(1.0, f"a{c}", 9, 10) for c in range(6)] + [(2.0, f"b{c}", 3, 10) for c in range(6)]

    (pair,) = _pairs(_check(rows))
    check = _check(rows)

    assert (pair.clusters_reversed, pair.clusters_ordered, pair.clusters_tied) == (0, 0, 0)
    assert pair.p_value == 1.0
    assert pair.flagged is False
    assert _contains(check.warnings, _NO_SHARED_WORDS)


def test_cluster_sign_test_requires_cluster_ids() -> None:
    rows = _hand_rows((1.0, 2.0), [[9, 5]] * 6)

    with pytest.raises(ValueError, match=r"(?i)cluster ids|obs\.cluster"):
        check_monotonicity(counts_obs(rows, clustered=False), dependence="clustered")


def test_cluster_sign_test_rejects_continuous_observations() -> None:
    from robustmargin import ContinuousObservations

    obs = ContinuousObservations.from_observations(
        LOG,
        severity=[1.0, 1.0, 2.0, 2.0],
        value=[1.0, 2.0, 1.5, 2.5],
        outcome="score",
        direction="decreasing",
        cluster=["a", "b", "a", "b"],
    )

    with pytest.raises(ValueError, match=r"(?i)continuous|binomial"):
        check_monotonicity(obs, dependence="clustered")


def _any_flagged(rows: list[Row]) -> bool:
    obs: Observations = counts_obs(rows)
    return check_monotonicity(obs, dependence="clustered", alpha=0.05).any_flagged


@pytest.mark.parametrize(
    ("label", "upper", "lower", "mu"),
    [
        ("falling", 1.0, 0.0, 1.8),  # monotone truth with a real trend
        ("flat", 0.6, 0.6, 1.8),  # no trend at all: every comparison is a fair coin
    ],
)
def test_cluster_sign_test_family_wise_false_flag_rate_does_not_exceed_alpha(
    label: str, upper: float, lower: float, mu: float
) -> None:
    """Under a monotone truth (and under equality, the boundary) the chance that any adjacent
    pair is flagged stays at or below 5%. Seeded; 300 simulated datasets of 24 balanced clusters
    over six levels. The assertion allows Monte Carlo error: it fails only if the exact 99% lower
    confidence bound of the observed rate exceeds alpha."""
    reps = 300
    flagged = 0
    for r in range(reps):
        rows = simulate_rows(
            20261400 + r,
            clusters=24,
            trials=10,
            sigma=1.0,
            levels=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
            upper=upper,
            lower=lower,
            mu=mu,
        )
        flagged += _any_flagged(rows)

    lower_bound = binomtest(flagged, reps).proportion_ci(confidence_level=0.99).low
    assert lower_bound <= 0.05, f"{label}: {flagged}/{reps} datasets flagged"
