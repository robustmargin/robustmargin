# robustmargin
Robustness margins with confidence intervals, via dose-response threshold estimation. Model-agnostic — no GPU, no ML framework required

> **Status: latest release `0.1.1`.** Binomial dose-response fits, thresholds with profile-likelihood or delta
> confidence intervals, censoring, ratios of two thresholds with Fieller or log-delta intervals,
> fit diagnostics, and the grid break-point rule. Continuous outcomes, clustered and paired
> inference, and design helpers are planned for `0.2`. While the version is below 1.0, a minor
> release may change the public API.

## What it does

You test a system at increasing levels of some stress (the *severity*) and count successes at
each level. robustmargin fits a dose-response curve to those counts and reports the severity at
which performance crosses a level you choose, with a confidence interval. When the data cannot
support a number, it says so: a threshold beyond the tested range is censored and reported as a
bound, and a failed fit carries a status flag and no estimate.

## Install

Requires Python 3.11 or later.

```bash
pip install robustmargin
```

Runtime dependencies are `numpy`, `scipy`, and `statsmodels`, with loose minimum versions.

## Worked example

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

Treat an overdispersed fit's intervals as too narrow. Clustered inference, which addresses this,
is planned for `0.2`.

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

The JSON follows `schema/scorecard-v1.json`, a JSON Schema shipped inside the package. It
covers binary outcomes only, and reading is strict: a card with fields this version does not know
is rejected, not silently trimmed. To read a stored card, use a robustmargin at least as new as the
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

## License

Apache-2.0. See [`LICENSE`](https://github.com/robustmargin/robustmargin/blob/main/LICENSE).
