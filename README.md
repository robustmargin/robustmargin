# robustmargin
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23585829.svg)](https://doi.org/10.5281/zenodo.23585829)

**Robustness margins for AI safety evaluation: how much stress an AI system takes before it
fails, with a confidence interval.**

robustmargin is a robustness calculation engine. AI systems are usually evaluated by a success
rate under nominal conditions; a safety case needs a different number: how far conditions can
degrade before the system starts to fail, and how sure we are of that point. robustmargin
estimates that robustness margin, with a confidence interval, from stress tests at increasing
severity, scored pass/fail or on a continuous scale.

It is model-agnostic and framework-free: it sees only severities and outcomes (success counts or
real-valued scores), and needs no GPU and no ML framework. That makes it usable wherever
robustness and safety carry weight:

- **AI safety research**: measure how much perturbation a model or policy tolerates, and whether
  a training intervention moved that margin.
- **Physical AI**: learned policies evaluated under graded sensing noise, latency or
  disturbances.
- **Autonomous and safety-critical systems**: margins reported with honest uncertainty, and
  censored bounds instead of invented numbers when the data cannot support an estimate.

> **Status: latest release `0.3.0`.** Dose-response fits for pass/fail outcomes that fall or rise
> with severity and for real-valued scores (Gaussian least squares), thresholds with
> profile-likelihood or delta confidence intervals, censoring, ratios of two thresholds with
> Fieller or log-delta intervals, fit diagnostics, and the grid break-point rule. The development
> version adds model-free isotonic fits, bootstrap intervals, and clustered and paired inference
> (see below). Design helpers are planned for `0.4`. While the version is below 1.0, a minor
> release may change the public API.

## What it does

You test a system at increasing levels of some stress (the *severity*), such as input noise, a
perturbation, a disturbance or a distribution shift, and count successes at each level (or record
a score for each run). robustmargin fits a dose-response curve to those outcomes and reports the
severity at which performance crosses a level you choose, with a confidence interval. When the
data cannot support a number, it says so: a threshold beyond the tested range is censored and
reported as a bound, and a failed fit carries a status flag and no estimate.

The statistics are dose-response threshold estimation, the method toxicology and psychophysics
use to find the dose at which a response crosses a given level. That is why the second worked
example below uses a classic toxicology dataset: the code is the same whatever the system under
test.

For control engineers, the threshold is an empirical analogue of the probabilistic robustness
margin of randomized robust control (Barmish, Lagoa & Tempo, ACC 1997; Chen, Zhou & Aravena,
SIAM J. Control Optim. 2008). That margin is the largest radius at which a known model still
succeeds with probability 1 − ε, averaged over every perturbation up to that radius.
robustmargin needs no model: it estimates the corresponding point from success counts at held
severities, with a confidence interval.

## How it fits into an evaluation stack

robustmargin does not run models. Your policy or model evaluator does, and a thin adapter turns
its results into robustmargin's input:

1. The evaluator applies a graded degradation (input noise, a perturbation, a disturbance, a
   distribution shift) at a set of severity levels.
2. It records the outcome of each run: pass/fail, or a real-valued score.
3. The adapter hands robustmargin the severities and the success counts or scores; robustmargin
   returns the threshold, its confidence interval, its censoring status and fit diagnostics.

Because the interface is only severities and outcomes, the same adapter pattern works for any
evaluator. A typical use is comparing checkpoints, for example before and after RL
post-training: `ratio_interval` gives a confidence interval for the ratio of their margins, so
"the new checkpoint is more robust" becomes a claim with an interval attached.

## Install

Requires Python 3.11 or later.

```bash
pip install robustmargin
```

Runtime dependencies are `numpy`, `scipy`, and `statsmodels`, with loose minimum versions.

## Worked example: robustness before and after RL post-training

Two checkpoints of the same policy, before and after RL post-training, are evaluated on a task at
increasing levels of Gaussian input noise: 200 runs per level, each scored pass or fail. The counts
are illustrative. The question a safety case asks is not "what is the success rate?" but "how much
noise can each checkpoint take before success falls below 90%, and did post-training move that
point?"

```python
from robustmargin import (
    Axis,
    Censoring,
    Definition,
    Observations,
    Status,
    fit_dose_response,
    ratio_interval,
    threshold,
)

axis = Axis(name="input_noise", unit="std", scale="log")
noise = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
runs = [200] * len(noise)
passed = {
    "before_rl": [196, 193, 181, 142, 61, 9],
    "after_rl": [197, 196, 192, 177, 119, 38],
}

margins = {}
for name, successes in passed.items():
    obs = Observations.from_counts(
        axis,
        severity=noise,
        successes=successes,
        trials=runs,
        outcome="task_success",
        direction="decreasing",
    )
    # Clean-input success is below 100%, so the upper asymptote is estimated, not fixed at 1.
    fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)
    assert fit.status is Status.OK
    margins[name] = threshold(
        fit, definition=Definition.absolute(0.9), interval_method="profile", level=0.95
    )

# The noise level at which success falls to 90%, with a 95% profile-likelihood interval.
before, after = margins["before_rl"], margins["after_rl"]
assert before.censoring is Censoring.NONE and after.censoring is Censoring.NONE
assert 0.096 < before.lo < before.value < before.hi < 0.130  # about 0.11
assert 0.159 < after.lo < after.value < after.hi < 0.217  # about 0.19

# Did RL post-training move the margin? The ratio of the two margins, with its own interval.
ratio = ratio_interval(after, before, method="log_delta", dependence="independent")
assert 1.35 < ratio.lo < ratio.estimate < ratio.hi < 2.06  # about 1.67x, interval excludes 1
```

When the data cannot support a number, robustmargin does not invent one. A checkpoint that never
drops below the criterion within the tested range gets a censored bound, not an estimate:

```python
from robustmargin import (
    Axis,
    Censoring,
    Definition,
    Observations,
    fit_dose_response,
    threshold,
)

obs = Observations.from_counts(
    Axis(name="input_noise", unit="std", scale="log"),
    severity=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8],
    successes=[200, 200, 199, 200, 199, 198],
    trials=[200] * 6,
    outcome="task_success",
    direction="decreasing",
)
fit = fit_dose_response(obs, model="binomial", link="probit", upper="estimate", lower=0.0)
t = threshold(fit, definition=Definition.absolute(0.9), interval_method="profile", level=0.95)

# "The margin exceeds 0.8" (exact one-sided bound), never "the margin is 0.8".
assert t.censoring is Censoring.RIGHT
assert t.value is None and t.lo == 0.8
```

## Worked example: a classic dose-response dataset

Finney's (1971) toxicity data: groups of insects exposed to increasing doses, with the number
affected in each group. The outcome here is survival, which decreases with dose. The dose-0 group
is an untreated control.

```python
from robustmargin import (
    Axis,
    Censoring,
    Definition,
    Observations,
    Status,
    fit_dose_response,
    threshold,
)

dose = [0.0, 2.6, 3.8, 5.1, 7.7, 10.2]
affected = [0, 6, 16, 24, 42, 44]
total = [49, 50, 48, 46, 49, 50]

obs = Observations.from_counts(
    Axis(name="dose", unit="mg", scale="log"),
    severity=dose,
    successes=[n - a for n, a in zip(total, affected, strict=True)],
    trials=total,
    outcome="survival",
    direction="decreasing",
)

# The link and the asymptotes are chosen before looking at the data; nothing is auto-selected.
fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower=0.0)
assert fit.status is Status.OK

# The dose at which survival falls to 50%, with a 95% profile-likelihood interval.
t50 = threshold(fit, definition=Definition.absolute(0.5), interval_method="profile", level=0.95)
assert t50.censoring is Censoring.NONE
assert abs(t50.value - 4.85) < 0.01
assert 4.3 < t50.lo < t50.value < t50.hi < 5.4

# A successful fit carries goodness-of-fit diagnostics, and every fit, failed or not, carries a
# monotonicity check on the raw counts. Both are reported and never acted on: nothing is
# rescaled, and the link you chose is never swapped for a better-scoring one.
diag = fit.diagnostics
assert diag.df == 3  # five dosed groups, two estimated parameters
assert diag.dispersion < 1.5  # no sign of overdispersion
assert set(diag.link_aic) == {"probit", "logit", "cloglog"}
assert not fit.monotonicity.any_flagged  # survival never rises between adjacent doses
```

### Comparing two curves

Two curves from the `selenium` data: survival against concentration for two forms of a
compound. `baseline_fraction(0.5)` asks where survival falls to half of what the control group
achieves, and the upper asymptote is estimated rather than fixed at 1. The second form shows
what the diagnostics are for.

```python
from robustmargin import (
    Axis,
    Definition,
    Observations,
    fit_dose_response,
    ratio_interval,
    threshold,
)

conc = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]
curves = {
    "form_a": ([151, 146, 116, 159, 150, 140], [3, 40, 31, 85, 102, 112]),
    "form_b": ([141, 153, 142, 139, 154, 155], [2, 30, 59, 82, 62, 85]),
}

thresholds = {}
fits = {}
for name, (total, dead) in curves.items():
    obs = Observations.from_counts(
        Axis(name="concentration", unit="ug/L", scale="log"),
        severity=conc,
        successes=[n - d for n, d in zip(total, dead, strict=True)],
        trials=total,
        outcome="survival",
        direction="decreasing",
    )
    fits[name] = fit_dose_response(
        obs, model="binomial", link="probit", upper="estimate", lower=0.0
    )
    thresholds[name] = threshold(
        fits[name],
        definition=Definition.baseline_fraction(0.5),
        interval_method="profile",
        level=0.95,
    )

ratio = ratio_interval(
    thresholds["form_a"], thresholds["form_b"], method="fieller", dependence="independent"
)
assert 0.5 < ratio.lo < ratio.estimate < ratio.hi < 0.9  # form_a fails at a lower concentration

# form_b is where diagnostics earn their place: survival rises from 300 to 400, more than chance
# allows, and the groups vary far more than a binomial model expects. Both are reported on the
# fit. The reversal is also repeated as a warning on every threshold and ratio built from it.
# Neither changes a number.
form_b = fits["form_b"]
flagged = [(p.severity_low, p.severity_high) for p in form_b.monotonicity.pairs if p.flagged]
assert flagged == [(300.0, 400.0)]
assert form_b.diagnostics.dispersion > 1.5
assert any(w.startswith("OVERDISPERSED: ") for w in form_b.diagnostics.warnings)
assert any(w.startswith("NON_MONOTONE_DATA: ") for w in ratio.warnings)
```

Treat an overdispersed fit's intervals as too narrow. When the extra variation comes from trials
that share a cluster, see "Repeated scenarios: clustered and paired inference" below.

### An outcome that rises with severity

Often what you count gets *worse* as severity grows: events per trial, not successes. Declare it
with `direction="increasing"`, and `successes` counts the outcome that rises. The curve runs from
`lower`, the rate at zero severity, up to `upper`. An increasing fit is the decreasing fit of the
complementary outcome (`trials - successes`) mapped back, so every status, interval and censoring
rule is the one the decreasing case uses, and every number and message is on the scale of the
outcome you counted.

`Definition.baseline_multiple(k)` asks where the rate reaches `k` times its value at zero
severity. It is defined on log axes only, and `threshold` raises `ValueError` on a linear axis. A
multiple of zero is not a threshold, so `lower` must be estimated (or fixed above zero): a fixed
`lower=0.0` over a control that has events is reported as `CONTROL_INCOMPATIBLE`, and
`baseline_multiple` raises for a fixed `lower` of zero. It also raises `ValueError` on any failed
fit (`SEPARATION` or `NOT_CONVERGED`), because a multiple of the control's observed rate would be
an uncovered target; `baseline_fraction(p)` raises on a failed fit for the same reason. Use
`Definition.absolute(a)` then, or a converged refit (for example with `lower` fixed above zero),
and catch the `ValueError` per job in a sweep.

```python
from robustmargin import (
    Axis,
    Definition,
    Observations,
    Status,
    fit_dose_response,
    threshold,
)

severity = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
events = [20, 22, 35, 90, 220, 340, 380]

obs = Observations.from_counts(
    Axis(name="severity", unit="unit", scale="log"),
    severity=severity,
    successes=events,
    trials=[400] * len(severity),
    outcome="event",
    direction="increasing",
)
fit = fit_dose_response(obs, model="binomial", link="probit", upper=1.0, lower="estimate")
assert fit.status is Status.OK
assert 0.03 < fit.params["lower"].value < 0.07  # about 5% of trials have an event at severity 0

# The severity at which the event rate doubles, with a 95% profile-likelihood interval.
doubling = threshold(
    fit, definition=Definition.baseline_multiple(2.0), interval_method="profile", level=0.95
)
assert doubling.status is Status.OK
assert 0.7 < doubling.lo < doubling.value < doubling.hi < 1.5
assert abs(fit.predict(doubling.value).estimate - 2 * fit.params["lower"].value) < 1e-6
```

`Definition.absolute(a)` and `Definition.relative(p)` mean the same in both directions, with the
crossing read upward. `Definition.baseline_fraction(p)` belongs to a decreasing outcome, and
`threshold` raises if you pair it with an increasing fit (and `baseline_multiple` with a
decreasing one). For an increasing outcome, `RIGHT` reads "still passes at the highest level
tested": the rate is confidently below the target there.

### A continuous response

When each run gives a real-valued score instead of a pass or fail, build
`ContinuousObservations` and fit with `model="gaussian"`. The curve is the same, now with
independent normal errors fitted by least squares, and `link` still names the shape of the
transition (`"logit"`, `"probit"` and `"cloglog"` are R `drc`'s `LL.4`, `LN.4` and `W1.4`).
`ContinuousObservations.from_observations` takes one response per observation and `from_summaries` takes
a mean, a sample SD and a count per level; both describe the same data and give the same fit.

`upper` and `lower` are in the units of the response, so a threshold target such as
`Definition.absolute(0.7)` is too. Intervals are calibrated on the fit's residual degrees of
freedom: the delta interval uses a Student-t quantile and the profile uses the F distribution on
the change in the residual sum of squares. A continuous response has no exact bounds, so a
threshold is `OK`, `UNREACHABLE` (the target lies outside the fitted range of the response) or
`NOT_CONVERGED`, and a value beyond the tested range carries an `EXTRAPOLATED:` warning. A fit that
cannot be identified (a transition wider than the tested range, a step between two levels, no
residual degrees of freedom, a response that does not move the way `direction` says) is
`NOT_CONVERGED` with the reason in `warnings` and no number.

```python
from robustmargin import (
    Axis,
    ContinuousObservations,
    Definition,
    Status,
    fit_dose_response,
    threshold,
)

severity = [x for x in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0) for _ in range(3)]
score = [
    0.95, 0.96, 0.94, 0.87, 0.88, 0.87, 0.80, 0.84, 0.78, 0.59, 0.63, 0.62,
    0.39, 0.36, 0.39, 0.27, 0.20, 0.23, 0.13, 0.14, 0.13,
]  # fmt: skip

obs = ContinuousObservations.from_observations(
    Axis(name="severity", unit="unit", scale="log"),
    severity=severity,
    value=score,
    outcome="score",
    direction="decreasing",
)
fit = fit_dose_response(obs, model="gaussian", link="logit", upper="estimate", lower="estimate")
assert fit.status is Status.OK
assert fit.residual.df == 17  # 21 observations, 4 estimated parameters

# The severity at which the score has fallen to 0.7, with a 95% profile interval.
margin = threshold(fit, definition=Definition.absolute(0.7), interval_method="profile", level=0.95)
assert margin.status is Status.OK
assert 0.6 < margin.lo < margin.value < margin.hi < 0.9
assert abs(fit.predict(margin.value).estimate - 0.7) < 1e-6
```

A ratio of two continuous thresholds uses a Student-t quantile on Welch-Satterthwaite degrees of
freedom, and `ratio_interval` raises `ValueError` for a count threshold against a continuous one.

### A model-free fit and bootstrap intervals

When you would rather assume only that performance falls (or rises) with severity, fit
`model="isotonic"`: it takes counts or a continuous response, needs no `link`, `upper` or `lower`,
and returns the monotone least-squares value of every tested level in `Fit.cell_estimates` (found
by pool-adjacent-violators). There is no curve between tested levels, so `Fit.predict` and
`diagnose` raise. A threshold on an isotonic fit is where the straight line drawn through those
values (on `log` severity for a log axis) first reaches the target. Nothing is extrapolated: a
target the line never reaches inside the tested levels is `UNREACHABLE` with no number. For counts
the exact bounds (`RIGHT`, `LEFT`, `FAILS_AT_BASELINE`) still come first. `relative` raises,
because there are no asymptotes, and `baseline_fraction` and `baseline_multiple` use the isotonic
value at the zero-severity control as the baseline.

An isotonic fit has no covariance, so its interval is a bootstrap: `interval_method="cell_bootstrap"`
resamples each tested level from its own data, refits, and reports percentiles of the resampled
thresholds. `n_resamples` defaults to 999 and `seed` makes the run reproducible; when you give no
seed one is drawn and recorded in `Threshold.bootstrap`, so any result can be rerun. A resampled
curve that never reaches the target counts toward an open side of the interval, so the interval
can be one-sided, or carry no bounds when the data cannot say either way.

```python
from robustmargin import Axis, Definition, Observations, Status, fit_dose_response, threshold

obs = Observations.from_counts(
    Axis(name="severity", unit="unit", scale="log"),
    severity=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
    successes=[100, 98, 90, 60, 20, 5],
    trials=[100] * 6,
    outcome="success",
    direction="decreasing",
)
fit = fit_dose_response(obs, model="isotonic")
assert fit.cell_estimates == (1.0, 0.98, 0.9, 0.6, 0.2, 0.05)  # already monotone

# The severity at which success falls to 50%, with a 95% bootstrap interval.
margin = threshold(
    fit,
    definition=Definition.absolute(0.5),
    interval_method="cell_bootstrap",
    level=0.95,
    n_resamples=499,
    seed=7,
)
assert margin.status is Status.OK
assert 2.0 < margin.value < 4.0
assert margin.lo <= margin.hi
assert margin.bootstrap.seed == 7 and margin.bootstrap.failed == 0
```

`interval_method="parametric_bootstrap"` does the same for a binomial or Gaussian fit: it simulates
new data from the fitted curve at the tested levels and refits the same model. On a fit with an
estimated asymptote each refit is far slower than with fixed asymptotes, so give it a smaller
`n_resamples`. A replicate whose refit fails is counted in `Threshold.bootstrap` and enters the
percentiles as the worst case for each end (so failures can only widen the interval), with a warning
above 1% and no interval above 10% (or above `alpha/2`, if that is smaller).

### Repeated scenarios: clustered and paired inference

When many trials come from the same scenario, they are not independent, and intervals that assume
they are come out too narrow. Give `Observations` a `cluster` id for every row and ask for
`dependence="clustered"`. The point estimate is the usual maximum-likelihood fit, unchanged; its
covariance becomes a cluster-robust sandwich that uses only the variation between clusters, and
`Fit.cluster_cells` keeps the per-cluster counts so that whole clusters can be resampled. This is
for count outcomes (`model="binomial"`) only. A clustered fit needs more clusters than it has
estimated parameters (`ValueError` otherwise), and a result built on few clusters carries a
`FEW_CLUSTERS:` warning stating the count it considers few. Monotonicity is checked by an exact sign
test across clusters (`check_monotonicity(obs, dependence="clustered")`).

A clustered threshold makes no exact claim, because the exact bounds treat trials as independent.
Its recommended interval is `"delta"`: the sandwich covariance with a Student-t quantile on
`G - 1` degrees of freedom. `"cluster_bootstrap"` (resample whole clusters with replacement and
refit) is an alternative that needs about 40 or more clusters, because a percentile interval from
few clusters covers less than its stated level; the other methods raise. A beyond-range crossing is
an `EXTRAPOLATED` value, not a `RIGHT` bound. When two outcomes are measured on the same clusters,
the ratio of their thresholds takes `dependence="paired"`, with `method="log_delta"` (or
`"fieller"`), which keeps the correlation between the two. The example below uses 12 clusters, so
its results carry the `FEW_CLUSTERS:` warning, which is raised below 20 clusters (below 40 for
`"cluster_bootstrap"`).

```python
import math

import numpy as np

from robustmargin import (
    Axis,
    Definition,
    Observations,
    Status,
    fit_dose_response,
    ratio_interval,
    threshold,
)

axis = Axis(name="severity", unit="unit", scale="log")
levels = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
n_clusters, trials = 12, 20
rng = np.random.default_rng(3)
scenario_shift = rng.normal(0.0, 0.4, n_clusters)  # each scenario is easier or harder


def observe(log_midpoint_shift: float) -> Observations:
    severity, successes, cluster = [], [], []
    for g in range(n_clusters):
        for x in levels:
            z = (math.log(x) - (math.log(6.0) + log_midpoint_shift)) / 0.6 if x > 0 else -4.0
            p = 1.0 - 0.5 * (1.0 + math.erf((z + scenario_shift[g]) / math.sqrt(2.0)))
            severity.append(x)
            successes.append(int(rng.binomial(trials, p)))
            cluster.append(f"scenario-{g}")
    return Observations.from_counts(
        axis,
        severity=severity,
        successes=successes,
        trials=[trials] * len(severity),
        outcome="success",
        direction="decreasing",
        cluster=cluster,
    )


margins = {}
for name, shift in {"a": 0.0, "b": 0.5}.items():
    fit = fit_dose_response(
        observe(shift),
        model="binomial",
        link="probit",
        upper=1.0,
        lower=0.0,
        dependence="clustered",
    )
    assert fit.status is Status.OK and fit.dependence == "clustered"
    margins[name] = threshold(
        fit,
        definition=Definition.absolute(0.5),
        interval_method="delta",
        dependence="clustered",
        level=0.95,
    )
    assert margins[name].status is Status.OK
    assert margins[name].lo < margins[name].value < margins[name].hi

# Outcome "b" is measured on the same twelve scenarios as "a": a paired ratio of the two margins.
ratio = ratio_interval(
    margins["b"],
    margins["a"],
    method="log_delta",
    dependence="paired",
    level=0.95,
)
assert ratio.status is Status.OK and ratio.dependence == "paired"
assert 1.0 < ratio.lo < ratio.estimate < ratio.hi
```

`method="fieller"` and `"log_delta"` with `dependence="paired"` use the covariance between the two
fits' cluster-robust estimates and Student-t quantiles on `G - 1` degrees of freedom. Both
thresholds must come from clustered fits over the same set of cluster ids. `method="cluster_bootstrap"`
with `dependence="paired"` resamples the same clusters for both and is the alternative; it also
computes the sandwich standard error alongside, and a `CLUSTER_CROSS_CHECK:` warning says when the
two disagree by more than a stated tolerance. A ratio of clustered thresholds on disjoint sets of
clusters (`dependence="independent"`) uses a Student-t quantile with Welch-Satterthwaite degrees of
freedom.

## Grid break-point rule

`grid_break_point` finds the smallest tested severity at which observed performance falls
strictly below a criterion. It reports an **observed grid statistic, not an estimated
threshold**. The reported value is always one of the severities you tested. It is never
interpolated, and it has no confidence interval.


```python
from robustmargin import Censoring, grid_break_point

severity = [0.0, 0.5, 1.0, 2.0, 4.0]
performance = [1.0, 0.98, 0.96, 0.91, 0.40]  # e.g. success rate at each level

result = grid_break_point(severity, performance, criterion=0.95)
assert result.value == 2.0  # first tested level strictly below 0.95
assert result.max_tested == 4.0
assert result.censoring is Censoring.NONE
```

When no tested level fails, there is no value. The result is right-censored and records how far
testing went, so it reads as "held up to 2.0", never as "cannot fail":

```python
from robustmargin import Censoring, grid_break_point

held = grid_break_point([0.0, 1.0, 2.0], [1.0, 0.99, 0.97], criterion=0.95)
assert held.value is None
assert held.max_tested == 2.0
assert held.censoring is Censoring.RIGHT
```

On a signed axis, pass `signed=True`. The rule then breaks on magnitude, on whichever side fails
first:

```python
from robustmargin import grid_break_point

signed = grid_break_point(
    [-0.2, -0.1, 0.0, 0.1, 0.2],
    [0.05, 0.77, 0.99, 0.33, 0.0],
    criterion=0.95,
    signed=True,
)
assert signed.value == 0.1
assert signed.max_tested == 0.2
```

Some points about the rule:

- A level sitting exactly on the criterion passes. Only `performance < criterion` fails.
- Monotonicity is neither assumed nor checked, and nothing is pooled or fitted.
- Invalid input raises `ValueError` rather than returning a placeholder. That covers empty or
  mismatched arrays, non-finite values, and negative severities without `signed=True`.

## Results and JSON

Every result is a frozen dataclass carrying a `schema_version` and a `provenance` mapping.
`provenance` records where the inputs came from; robustmargin stores it and never interprets it.

- `GridBreakPoint` is what `grid_break_point` returns.
- `Fit`, `Threshold` and `Ratio` are what `fit_dose_response`, `threshold` and `ratio_interval`
  return. They check their own consistency when constructed. For example, a failed fit cannot
  carry parameters, and a censored threshold cannot carry a point estimate.

Results are collected in a `Scorecard` and written to JSON with `robustmargin.report`:

```python
import json

from robustmargin import Scorecard, grid_break_point
from robustmargin.report import from_dict, load_schema, to_dict

held = grid_break_point([0.0, 1.0, 2.0], [1.0, 0.99, 0.97], criterion=0.95)
card = Scorecard(results=(held,), provenance={"run": "example"})

text = json.dumps(to_dict(card), allow_nan=False)
assert from_dict(json.loads(text)) == card
assert load_schema()["$schema"] == "https://json-schema.org/draft/2020-12/schema"
```

The JSON follows `schema/scorecard-v3.json`, a JSON Schema shipped inside the package, and
`schema/scorecard-v1.json` and `schema/scorecard-v2.json` are kept so that cards written by
earlier releases still read, and are written back as `"3"`. Schema `"2"` added increasing outcomes,
`baseline_multiple` and the Gaussian model; schema `"3"` adds the isotonic model, the two bootstrap
interval methods, `Fit.cell_estimates` and `Threshold.bootstrap`. Reading is strict: a card with fields this version does not know is rejected, not silently
trimmed. To read a stored card, use a robustmargin at least as new as the
one that wrote it. The card's `library_version` field records which version that was.

For your own tests, `robustmargin.testing` provides `fake_fit`, `fake_threshold` and `fake_ratio`.
They return schema-valid results in every status and censoring state, and each is labelled as a
fake rather than an estimate.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy
pytest            # enforces 90% branch coverage
```

CI runs the tests on Python 3.11, 3.12, and 3.13 against current dependencies. It also runs them
on Python 3.11 against the oldest supported versions, pinned in
[`ci/constraints-min.txt`](https://github.com/robustmargin/robustmargin/blob/main/ci/constraints-min.txt).
Changes are recorded in [`CHANGELOG.md`](https://github.com/robustmargin/robustmargin/blob/main/CHANGELOG.md).

## Citing

robustmargin is written and maintained by Babak Badkoubeh
([ORCID profile](https://orcid.org/0009-0005-3586-7901)). If you use it in published work, please
cite it. Each release is archived on Zenodo
([doi:10.5281/zenodo.23585829](https://doi.org/10.5281/zenodo.23585829)), and the citation
metadata is in
[`CITATION.cff`](https://github.com/robustmargin/robustmargin/blob/main/CITATION.cff); GitHub's
"Cite this repository" button renders it as BibTeX or APA.

## License

Apache-2.0, copyright 2026 Babak Badkoubeh. See
[`LICENSE`](https://github.com/robustmargin/robustmargin/blob/main/LICENSE) and
[`NOTICE`](https://github.com/robustmargin/robustmargin/blob/main/NOTICE).
