# Changelog

All notable changes to robustmargin are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html). While the version is below 1.0, a
minor release may change the public API, and every such change carries a migration note.

## [Unreleased]

## [0.3.0] — 2026-11-11

### Added
- **Outcomes that rise with severity.** `Observations(direction="increasing")` now fits and
  thresholds like any other. `successes` counts an outcome that rises with severity, such as an
  event rate: `lower` is the rate at zero severity and `upper` the rate it approaches. It is fit
  as the decreasing curve of the complementary outcome (`trials - successes`, asymptotes
  `1 - lower` and `1 - upper`) and mapped back, so the likelihood, the optimisers, the censoring
  and exact-bound rules and both interval methods are the ones the decreasing case uses.
  - `Fit.params`, `Fit.covariance`, `Fit.predict` and its band, `Threshold.baseline`, and every
    warning and error message are on the scale of the outcome as counted. `Fit.cells` holds the
    counts as given, and `Fit.monotonicity` is the check on them.
  - `CONTROL_INCOMPATIBLE` for an increasing outcome is a `lower` fixed at `0` over a control with
    events, and its message suggests `lower="estimate"`.
  - For an increasing fit, `RIGHT` reads "still passes at the highest level tested" (the rate is
    confidently below the target there), and `LEFT` and `FAILS_AT_BASELINE` read the same way
    round. `absolute(a)` crosses `a` upward and `relative(p)` is unchanged.
- **`Definition.baseline_multiple(k)`**, for increasing outcomes: the severity at which the rate
  reaches `k` times its value at zero severity (`k > 1`).
  - `baseline_fraction(p)` raises for an increasing fit, and `baseline_multiple(k)` for a
    decreasing one.
  - A baseline at or below zero raises `ValueError` (a multiple of zero is not a threshold): a
    converged fit whose `lower` is fixed at `0`. Estimate `lower` or fix it above zero.
  - It raises `ValueError` on any failed fit (`SEPARATION`, `NOT_CONVERGED`), naming the fit
    status and pointing to `absolute(a)`: a multiple of the control's observed rate is an
    estimate treated as fixed, and its exact claims would carry no coverage guarantee.
  - It is defined on log axes only: on a linear axis the lower asymptote is not the performance at
    zero severity, so `threshold` raises `ValueError`, and a `Threshold` built or read back with
    `baseline_multiple` on a linear axis is rejected.
  - `fit_dose_response` raises `ValueError` for an increasing outcome with a fixed `lower` that is
    not zero but so small that `1 - lower` equals 1 in double precision (for example `1e-17`).
  - `Threshold` itself refuses a `baseline_multiple` definition with a decreasing direction and a
    `baseline_fraction` definition with an increasing one, whether built directly, by
    `dataclasses.replace` or read back through `from_dict`.
- **Score-card schema `"2"`** (`schema/scorecard-v2.json`), which differs from `"1"` in
  admitting `"increasing"` for `Fit.direction` and `Threshold.direction`, `"baseline_multiple"`
  for `Definition.kind`, and Gaussian (continuous-response) fits (see below).
  `schema/scorecard-v1.json` is unchanged and `load_schema("1")` still loads it. `from_dict` reads
  both versions, and rejects a result tagged `"1"` that carries an increasing direction, a
  `baseline_multiple` definition or a Gaussian fit. The check is per object: it looks at each
  `Fit` or `Threshold` and its own `schema_version`, not at the card as a whole.
- `robustmargin.testing.fake_fit` and `fake_threshold` take a keyword-only `direction`
  (`"decreasing"` by default, so existing calls are unchanged); `"increasing"` builds a schema-valid
  increasing-outcome result.
- `ratio_interval` accepts two thresholds of different directions on the same axis and definition;
  the ratio of severities does not depend on which way each outcome runs.
- **Continuous responses: `model="gaussian"`.** `fit_dose_response` fits the same four-parameter
  curve to real-valued responses by nonlinear least squares with independent normal errors
  (`ContinuousObservations`), and `threshold` and `ratio_interval` work on the result. `logit`,
  `probit` and `cloglog` are R `drc`'s `LL.4`, `LN.4` and `W1.4`.
  - **Data.** `ContinuousObservations.from_observations` takes one response per observation and
    `from_summaries` a mean, a sample SD (`ddof=1`) and a count per level; the two give the same
    fit. `ContinuousCell` is a pooled level, and `cells()` pools rows at one severity accurately
    where the sum-of-squares shortcut loses every digit. `fit_dose_response` raises `ValueError`
    for `ContinuousObservations` with `model="binomial"` and for `Observations` with
    `model="gaussian"`.
  - **Fit.** `Fit.residual` (new, `Residual(sigma, df, rss)`) is set exactly for an `OK` Gaussian
    fit and is `None` for a binomial one. The covariance is the residual variance `rss / df` times
    the inverse of the observed Hessian of the residual sum of squares on the natural parameters,
    computed analytically.
    `upper` and `lower` are any finite response values with `upper > lower` (a fixed one costs no
    degree of freedom). An increasing response is the exact negation of the decreasing fit of the
    negated response, so the two agree bit for bit. `Fit.cells` holds `ContinuousCell`,
    `Fit.diagnostics` is a `ContinuousDiagnostics` (residual sum of squares, a pure-error
    lack-of-fit F test when levels are replicated, and the AIC of the three links), and
    `Fit.monotonicity` is a one-sided Welch test between adjacent levels, Holm-corrected
    (`ContinuousAdjacentPair`). **These are now unions.** `Fit.cells`, `Fit.diagnostics`,
    `diagnose()` and `MonotonicityCheck.pairs` hold the binomial or the continuous type by
    `Fit.model`, so a caller that reads `.successes`, `.deviance` or `.successes_low` needs an
    `isinstance` check (or a test of `fit.model`) first.
  - **Failure is a flag.** A Gaussian fit that cannot be identified is `NOT_CONVERGED` with no
    parameter, covariance, likelihood, residual or diagnostics, and one of these reasons in
    `warnings`: `df`, `boundary`, `flat`, `step`, `singular`, `optimizer` or `zero residual`. The unidentified cases are one test: the fit is no better than a supremum of the closure of
    the curve family, whatever its scale. Responses are unbounded, so a free asymptote can run to
    infinity, and the suprema are the `s -> 0` step (`step`), the `s -> infinity` line or constant
    (`flat`) and the exponential tails `a - A exp(lam t)` and `a + A exp(-lam t)` (`unbounded
    asymptote`). The same limits apply when no start converged. There
    is no separation, no exact bound and no fallback to another model.
  - **Intervals are calibrated on the residual degrees of freedom.** The delta interval is
    `theta ± t(df) · se` (on the log scale, so positive on every axis), and the profile interval
    uses `(RSS(theta) − RSS) / (RSS / df)` against `F(1, df)`. `Fit.predict` gives a Student-t band
    on the response scale, not clipped to the unit interval.
  - **No exact claims.** `FAILS_AT_BASELINE`, `RIGHT` and `LEFT` are never produced. A Gaussian
    threshold is `OK`, `UNREACHABLE` (the target is not strictly inside the fitted `(lower,
    upper)`) or `NOT_CONVERGED` (no value, bound or grid). A value outside the tested range carries
    an `EXTRAPOLATED:` warning, and an unclosed profile side is `OPEN_UPPER` or `OPEN_LOWER`
    (the delta interval is built on the log scale on every axis, so it is positive).
    `absolute(a)` takes any finite `a` in response units; `baseline_fraction(p)` raises when the
    fitted `upper` is not positive and `baseline_multiple(k)` when the fitted `lower` is not.
  - **Ratios.** Two Gaussian thresholds use a Student-t quantile on Welch-Satterthwaite degrees of
    freedom: `log_delta` at the degrees of freedom of `Va + Vb`, and `fieller` at those of its
    pivot, evaluated at the ratio estimate. A count threshold and a continuous one are not
    comparable, and `ratio_interval` raises `ValueError`.
  - **Delta standard error for `absolute(a)`.** It uses the full delta method over every free
    parameter, the asymptotes included. R `drc`'s `ED(type="absolute")` converts `a` to a relative
    level at the fitted asymptotes and holds it fixed, so its standard error can differ (about 20%
    on the ryegrass data). Point estimates agree with `drc`.
  - **Schema and fakes.** Schema `"2"` gains `ContinuousCell`, `Residual`, `ContinuousDiagnostics`
    and `ContinuousAdjacentPair`, `Fit.model` accepts `"gaussian"`, and the shape of `cells`,
    `diagnostics`, `residual` and the bounds on `upper` and `lower` depend on the model. A binomial
    `Fit` writes no `residual` key, so its serialised form is unchanged.
    `robustmargin.testing.fake_fit` and `fake_threshold` take a keyword-only `model` (`"binomial"`
    by default) and build Gaussian results with `model="gaussian"`.

### Changed
- **Breaking change: `baseline_fraction` refuses on a failed fit.** `threshold()` with
  `Definition.baseline_fraction(p)` on a fit whose status is not `OK` raises `ValueError`, as
  `baseline_multiple` does. It previously used the control's observed rate as a fixed target
  `p·k₀/n₀`, whose exact claims carry no coverage guarantee for the definition's own target
  `p·P(0)`. This includes the `UNREACHABLE` result a control with no events (`0/n`) used to give. Migration: state the target explicitly with `Definition.absolute(p * k0 / n0)` if the
  plug-in is what you want, or refit so the fit converges.
- **Behavioural change: `direction="increasing"` no longer raises.** `fit_dose_response`, `Fit` and
  `Threshold` used to refuse an increasing binomial outcome; a call that raised now returns a
  result. Code that caught the `ValueError` to detect an unsupported direction no longer sees it.
- **Breaking change: `schema_version` is `"2"` on every result.** The default of
  `schema_version` on `Fit`, `Threshold`, `Ratio`, `GridBreakPoint` and `Scorecard` is now `"2"`,
  `report.SCHEMA_VERSION` is `"2"`, and `to_dict` writes `"2"` at every nesting level of a
  freshly built result, including one that uses nothing new. A result read back from a `"1"` card
  keeps `"1"` and writes `"1"` again. `report.load_schema()` defaults to `"2"`, so its `$id` and
  title are now the v2 document's.
  - **Migration: upgrade readers before writers.** 0.1.1 and 0.2.0 read schema `"1"` only. They
    reject every card this release writes with a `ValueError` naming the unknown version, even a
    card with nothing new in it and even a bare `GridBreakPoint`. Upgrade every reader of a shared
    card store before anything writes to it with this release. Cards written by 0.1.1 or 0.2.0
    stay readable. Validating against `scorecard-v1.json` does not catch the mismatch, because the
    schema does not constrain `schema_version`.
  - **Migration: branch on direction and definition.** Code that reads stored results must branch
    on `Fit.direction` and `Threshold.direction`, and handle `Definition.kind ==
    "baseline_multiple"`. For `"increasing"`: `params["lower"]` is the zero-severity rate; `RIGHT`
    means the rate is confidently *below* the target at the highest level; `BaselineRate.successes`
    counts events; and `CONTROL_INCOMPATIBLE` means `lower` is fixed at 0 over a control with
    events.
  - **Compare `schema_version` with care.** Code that compares it literally against `"1"`
    (for example `result.schema_version == "1"` or a check on `asdict(grid_break_point(...))`)
    now sees `"2"`, and must accept both or compare against `report.SCHEMA_VERSION`. Reading a
    `"1"` card still yields objects whose `schema_version` is `"1"`.
  - A nested or bare result that omits `schema_version` is now read as `"2"`; only the
    `Scorecard` root requires the tag.
- **The profile interval's search shortens its stride when the inner fit fails.** When a stride
  lands where the inner constrained fit fails, the search now bisects that stride (at most 30
  times) instead of leaving the side open, so it finds an endpoint that lies before the failure.
  - Migration: a profile bound previously reported open with the "inner fit failed" warning may
    now be finite. `censoring` then goes from `OPEN_LOWER`/`OPEN_UPPER` to `NONE`, or, for an
    interval that was open on both sides (`NONE` with no bounds), to `OPEN_LOWER`/`OPEN_UPPER`.
    In that second case a `ratio_interval` built on the threshold now comes back censored with no
    numbers, as it does for any one-sided input. Results whose search never met a failed inner
    fit are unchanged.
- **Breaking change: `Threshold` rejects `baseline` and `grid` on the wrong status.** It now
  rejects `baseline` on any status other than `FAILS_AT_BASELINE`, and `grid` on any status other
  than `SEPARATION` and `NOT_CONVERGED`, including `UNREACHABLE`, `CONTROL_INCOMPATIBLE` and
  `FAILS_AT_BASELINE`, which previously skipped these checks. No released `threshold()` or
  `fake_threshold` produces such a result. A hand-built `Threshold` that does now raises
  `ValueError` at construction, and a hand-edited card fails `from_dict`. Migration: set the field
  to `None` (`null` in JSON).
- **Behavioural change: profile bounds at the reachability wall.** For `Definition.absolute(a)`
  with an estimated approaching asymptote, the profile interval's far bound can move outward
  where the constrained maximum lies at the reachability wall: the asymptote the curve approaches
  tends to the target while the transition stays inside the data. The inner fit now also starts
  from the curve refitted with that asymptote fixed at the target, placed at each candidate
  threshold. The start is only a starting point (the refit's own likelihood is never used), so
  the profile value is always one attained by a feasible curve.
  - The best scanned alternative, and the wall start where it is the best run overall, are
    Newton-polished and the profile log-likelihood is the largest of those values and the first
    run's, so it is never below what it was without the wall start. At every threshold the
    deviance is no higher than before, and the inner fit fails at no threshold where it succeeded
    before. The search steps through the same thresholds, so it brackets the crossing at the same
    step or further out. A bound can come out lower only where the deviance crosses the target
    more than once inside the last sub-interval, or where the previous search met an inner-fit
    failure, in which case a side it left open can now close.
  - The visible consequence, on designs whose constrained maximum lies at the wall: a finite
    upper bound with `censoring=NONE` becomes `hi=None` with `OPEN_UPPER`, because the deviance
    stays below the target up to the search limit. This is on the severity axis, so it reads the
    same for an increasing outcome. A `ratio_interval` built on such a threshold then comes back
    censored, as it does for any one-sided input.
  - A refit that raises an error (numerical or validation) adds a warning to the interval result,
    and the interval uses the other starts. A refit that ends with a status other than `OK` has no
    curve to start from and is silent.
  - **Migration:** nothing to change in calls. Code that assumed a closed upper bound for
    these thresholds must handle `hi=None` and `OPEN_UPPER`. Affected: an absolute target with
    an estimated approaching asymptote (the lower asymptote of the decreasing curve, the upper
    asymptote of an increasing outcome). Unchanged: a fixed approaching asymptote, `relative`,
    `baseline_fraction` and `baseline_multiple` targets.

### Fixed
- An increasing fit with `absolute(a)` for a tiny `a` (`0 < a <= 2**-54`) is still `UNREACHABLE`,
  but its warning now says that `1 - a` rounds to 1 in double precision, instead of claiming that
  the target is outside the open interval (0, 1).
- **Direct-likelihood fits could stop at a local maximum.** In 0.2.x this affects
  `fit_dose_response` with an estimated asymptote, or with a fixed one other than `upper=1`,
  `lower=0`. The optimiser ran from a single start, so it could return `OK` at a local maximum
  of the likelihood. In one reported design, a curve with log-likelihood −16.59 was reported
  `OK` while the maximum was −6.56. Every threshold, interval and ratio built on such a fit was
  wrong.
  - It could also report `NOT_CONVERGED` for data that do have a maximum.
  - The optimiser now also starts from five fixed points on the centred, scaled severity axis,
    and then from the best points of a deterministic scan of the log-likelihood. The scan covers
    the log scale from far below to far above the span of the tested levels; the curve's position
    at, between and beyond the tested levels, including the gap between a zero-severity control
    and the first nonzero level; and a grid of each estimated asymptote's range, which includes
    the asymptotes of the best step between tested levels. It keeps a better start only when
    that start beats the best log-likelihood found so far, and every convergence check then runs
    once on the start it kept. The method is deterministic, and a fit already at its maximum is
    unchanged by the extra starts. Direct-likelihood fits take about one and a half times as long
    as with the five fixed starts alone.
  - A fit whose log-likelihood does not exceed that of the best constant curve (`s → ∞`) is now
    `NOT_CONVERGED`, with the wrong-sign/flat reason. On a log axis a zero-severity control stays
    at `upper` on that curve.
  - Migration: a direct-likelihood fit may now report a higher log-likelihood with different
    parameters, turn from `NOT_CONVERGED` to `OK`, or turn from `OK` to `NOT_CONVERGED` when its
    supremum lies on a boundary. A `NOT_CONVERGED` fit's first warning may name a different
    reason; for example, a boundary refusal can become a step refusal when the scan finds a higher
    point on the step plateau. `diagnose`'s per-link refits use the same path, so their AIC and status can change
    too. Fits with both asymptotes fixed at `1` and `0` are unaffected, since that likelihood has
    a single maximum. The exception is a direct-likelihood refit that does not beat the
    constant-curve log-likelihood, which is now `NOT_CONVERGED`.
- **Profile intervals with an estimated asymptote could be too narrow.** This affects
  `threshold(..., interval_method="profile")` for a fit with an estimated asymptote, or a fixed
  one other than `upper=1`, `lower=0`. At each candidate severity the constrained fit ran from a
  single start. Where it stopped short of the constrained maximum, the
  deviance jumped above the chi-square target and the jump was reported as the bound, still `OK`.
  The same data could also give different bounds for an increasing outcome and its mirrored
  decreasing form.
  - The constrained fit now also starts from points chosen by deterministic scans of the
    log-likelihood: over the log scale crossed with the curve's position against the target, the
    free asymptote, and, where the target pins an asymptote at 0 or 1, that boundary. A start is
    kept only when it beats the original start's log-likelihood. Where the original start
    already puts the deviance below the target, the extra starts are skipped, since they cannot
    change which side of the target that point is on.
  - Once a sign change is bracketed, the profile is evaluated at interior points and the bound is
    solved in the first sub-interval that changes sign.
  - The inner fits may still miss the maximum in rare cases. In a perturbation search about 0.3%
    of bounds with an estimated asymptote were still too narrow, mostly upper bounds far beyond
    the largest tested severity.
  - `threshold` now raises `ValueError` when the constrained fit at the point estimate beats the
    fit's log-likelihood, which means the fit itself is not at its maximum.
  - Migration: a profile bound may move outward, may become open, and in rare cases moves inward.
    `censoring` can change from `NONE` to `OPEN_LOWER`/`OPEN_UPPER`. Point estimates are
    unchanged, and so are profile intervals with both asymptotes fixed at `1` and `0`. Profile
    intervals on the affected fits take about two to three times as long.

### Known limitations
- Profile bounds with an estimated asymptote can still be slightly too narrow in rare cases: about
  0.3% of interval sides in a perturbation search, mostly upper bounds far beyond the largest
  tested severity (see the local-maximum entry under Fixed).
- `Fit.predict` on a count fit with an estimated asymptote: at a severity far beyond the tested
  range, where the fitted curve equals its asymptote in double precision, `lo` and `hi` are `nan`;
  slightly closer in, they can collapse onto the estimate. This is unchanged from 0.2.x. The
  estimate itself is unaffected. Read the band within or near the tested range.

## [0.2.1] — 2026-10-23

### Fixed
- A fit with estimated asymptotes no longer raises when the optimizer steps to a point where
  the two asymptotes coincide in floating point.
- A fit by direct likelihood maximisation (estimated asymptotes, fixed asymptotes other than
  `1`/`0`, and the refit that replaces a GLM estimate) at the maximum is no longer reported
  `NOT_CONVERGED` because the optimizer's step-size criterion stalled at rounding level. The
  Newton decrement replaces only that step test; the flat-slope, boundary, regression, score and
  information checks still decide.
- **Behavioural change: fixed-asymptote fits (`upper=1`, `lower=0`) land on the true
  maximum-likelihood estimate, their profile intervals are computed from the true profile
  likelihood, and a non-finite log-likelihood no longer raises.**
  - **Before.**
    - The binomial log-likelihood formed each probability by subtraction, so in a tail it rounded
      to exactly `0` and the log-likelihood was `-inf`, even at the maximum; building the `Fit`
      then raised `ValueError`.
    - statsmodels' GLM reports convergence at points that are not the maximum (some
      complementary log-log fits, up to about 0.3 standard errors off in `mu`, rarely probit), and
      reports non-convergence on data whose maximum is finite and unique (a large share of
      complementary log-log fits with a rising tail).
    - Its observed information is not finite when a cell's fitted probability is exactly 0 or 1,
      which made some logit fits `NOT_CONVERGED`.
    - The profile interval's inner offset GLM had the same defect, so some complementary log-log
      and probit profile intervals (in 0.2.0 as well) were too narrow, or raised.
  - **Now.**
    - The log-likelihood is computed in log space, so it stays finite and accurate in the tails.
    - A GLM estimate is accepted only if the Newton decrement of that likelihood,
      `g' (-H)^-1 g / 2`, is at most `1e-6`: the log-likelihood a Newton step could still gain.
      The criterion does not depend on the parametrisation or the sample size.
    - When the GLM does not converge, fails that check, or has an observed information that cannot
      be evaluated, the same model is refit by direct likelihood maximisation with a statsmodels
      optimizer. The fit carries a warning naming which of the three happened ("the GLM did not
      converge", "failed the optimality check", "information could not be evaluated"). If only
      the information is non-finite and the estimate passes the check, the information is
      computed from the stable log-likelihood and the warning says so.
    - The profile's inner fit is checked the same way. Where it is replaced by direct constrained
      maximisation of the same model, `Threshold.warnings` says "the profile inner fit was
      replaced".
    - Some complementary log-log (rarely probit) fits move to the true maximum. Some
      `NOT_CONVERGED` fits become `OK`.
    - A fit whose log-likelihood is still not finite is `NOT_CONVERGED` with a warning, with no
      parameters, covariance or log-likelihood. It does not raise.
    - `diagnose`'s link comparison uses the same fitting path, so its comparison links inherit all
      of this: a comparison fit no longer takes down an `OK` primary fit.
    - `threshold()` on a fit that was `NOT_CONVERGED` and is now `OK` returns a model value and
      interval instead of the exact-bound fallback, for both interval methods.
  - **Migration.** Re-run analyses that use the complementary log-log link, that reported
    `NOT_CONVERGED`, or that used profile intervals on such fits. Code that caught `ValueError`
    around `fit_dose_response` for this case should check `fit.status` instead. Results where the
    GLM was already at the maximum are unchanged.
- A non-finite estimate or log-likelihood from the optimizer is `NOT_CONVERGED` with a warning
  saying the estimate is not finite, rather than being reported as `OK` or blamed on the
  information matrix.
- The optimality check's numerical derivatives (used for GLM estimates and the profile's inner
  fit) reduce their step to a fraction of the estimate's standard error where the default step
  is too coarse, so a true maximum of a steeply curved or badly scaled likelihood is no longer
  rejected by truncation error. Steps are never enlarged.
- **Behavioural change: a fit whose data are a step between adjacent tested levels is
  `NOT_CONVERGED`.**
  - **Before.** With an estimated asymptote or a fixed one other than `1`/`0`, such data have no
    finite maximum (the log-likelihood rises as the slope `s` tends to 0). Depending on the
    platform, the optimizer stopped at an arbitrary point and some fits were `OK` in 0.2.0, with
    very large standard errors.
  - **Now.** A fit is `NOT_CONVERGED` with the warning "no fitted curve exceeded the
    log-likelihood of a step between adjacent tested levels" when no stage of the optimizer
    exceeded the log-likelihood of the closed-form `s -> 0` step, with no parameters, covariance
    or log-likelihood. An estimated asymptote at a parameter-space boundary keeps its boundary
    reason, and flat or rising data keep the wrong-sign/flat reason. Some fits that were already
    `NOT_CONVERGED` now carry the step reason instead of the optimizer's. A fit that was `OK` in
    0.2.0 at a local maximum below the step's log-likelihood is now `NOT_CONVERGED` too.
  - **Migration.** Code that relied on an `OK` fit for step-like data should check `fit.status`.
    On a `NOT_CONVERGED` fit, `threshold()`'s exact bracket applies to definitions that do not
    need fitted asymptotes (`Definition.absolute`); a `relative` definition raises `ValueError`.

### Documented
- On a failed fit (`SEPARATION` or `NOT_CONVERGED`), `Definition.baseline_fraction(p)` resolves
  its target from the control's observed rate, `p * k0 / n0`. The exact claims are tests against
  that plug-in target and carry no coverage guarantee for `p` times the true zero-severity rate.
  For a covered claim on a failed fit, use `Definition.absolute(a)`. The `Threshold` and
  `baseline_fraction` docstrings say so.

## [0.2.0] — 2026-10-20

### Changed
- **Behavioural change: the exact results `threshold()` reports are now jointly correct at the
  stated confidence.**
  - **What is covered.** Every exact result is covered, of whatever kind: `FAILS_AT_BASELINE`,
    `RIGHT`, `LEFT`, and a failed fit's bracket or one-sided bound. The chance that the call
    returns an exact result that is wrong is now at most `1 - level` *in total*. Before, each kind
    was bounded on its own, and a stress design reached about 8% at `level=0.95`.
  - **Conditions.** The bound assumes a curve that does not increase with severity, independent
    trials, and a fixed target. The `Threshold` docstring states the limits:
    - When a tested severity's true rate equals the target exactly, the proved bound is
      `1.5 * (1 - level)`.
    - A target taken from the control's observed rate, as for a failed fit under
      `baseline_fraction`, is covered in simulation but not by proof.
    - Model-based values and intervals keep their own model-based coverage.
  - **How.** The exact claims share the `1 - level` error budget among four one-sided exact tests,
    each at confidence `1 - (1 - level) / 4` (0.9875 at 0.95):
    - an ascending run from the lowest nonzero tested severity;
    - a test of the highest severity;
    - a descending run from the highest severity, down to the lowest nonzero one;
    - a test of the lowest nonzero severity.

    The zero-severity control is not part of the runs. It is tested on its own, at the same
    confidence.
  - **Results can move on the same data.**
    - `RIGHT` and `LEFT` are reported less often. In a simulation with eight severities whose true
      rate is 0.95, against a target of 0.9, `RIGHT` fell from about 80% to 59% of calls.
    - A converged fit that no longer reports `RIGHT` or `LEFT` reports `(OK, NONE)` with the
      model's value instead. That value can lie far outside the tested range, and with
      `interval_method="profile"` it can come with `lo` and `hi` both `None`.
    - `FAILS_AT_BASELINE` is reported less often. It now needs the lowest nonzero severity to fail
      as well as the control. Such inputs now report `UNREACHABLE`, a model value or a failed-fit
      bound, with `baseline` set to `None`.
    - Failed-fit brackets and one-sided bounds can widen. A one-sided bound is no longer recomputed
      at `level`, and no bound is placed at the control.
    - `absolute(a)` with `a <= 0` or `a >= 1` is now `UNREACHABLE` (see below).
  - **Migration.**
    - Re-run any analysis that reads exact bounds, `RIGHT`/`LEFT` or `FAILS_AT_BASELINE`.
    - `value is None` no longer marks every result outside the tested range. Before using `value`
      from an uncensored `OK` result, check `Threshold.warnings` for `EXTRAPOLATED: ` (see Added).
      `Ratio.warnings` carries the token too.
- **Behavioural change: `UNREACHABLE` for impossible targets, on any fit status.**
  - A resolved target at or below 0, or at or above 1, is now `UNREACHABLE`, with a warning naming
    the target. This is checked before every exact test.
  - Before, such a target made an exact test trivially true or false. For example, a failed fit
    with a 0/n control under `baseline_fraction` returned `RIGHT`, and `absolute(a)` outside
    (0, 1) could return `FAILS_AT_BASELINE`.
  - **Migration.**
    - `UNREACHABLE` no longer implies `fit.status is OK`. On a failed fit, `Threshold.status` is
      `UNREACHABLE` rather than the fit's status, and `grid` is `None`. Detect a failed fit from
      `Threshold.fit.status`, not from `Threshold.status`.
    - `Threshold` now accepts `UNREACHABLE` on a fit whose status is `OK`, `SEPARATION` or
      `NOT_CONVERGED`. robustmargin 0.1.1 rejects such a stored result when reading it, so
      upgrade the code that reads results before the code that writes them.

### Added
- **`EXTRAPOLATED: ` warning token** (report-only).
  - It fires when a converged fit's reported value lies outside the range of tested nonzero
    severities. It changes no number.
  - It also appears on results whose numbers did not change in this release.
  - Match it with `w.startswith("EXTRAPOLATED: ")`.

### Fixed
- **Profile search with the estimate outside a search limit.** The profile-likelihood search no
  longer returns a bound on the wrong side of the estimate when the estimate lies at or beyond a
  search limit. That side is now reported as open.
- **Open-ended intervals checked on construction.**
  - `Threshold` rejects an `OPEN_UPPER` result whose `lo` exceeds `value`, and an `OPEN_LOWER`
    result whose `hi` is below it.
  - **Migration.** A stored result that fails this check has a bound on the wrong side of its
    value, from the profile-search defect above. Reading it now raises `ValueError`. Regenerate
    any result that fails this check from its inputs.

## [0.1.1] — 2026-10-17

First release.
