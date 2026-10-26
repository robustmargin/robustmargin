# Changelog

All notable changes to robustmargin are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html). While the version is below 1.0, a
minor release may change the public API, and every such change carries a migration note.

## [Unreleased]

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
- **Score-card schema `"2"`** (`schema/scorecard-v2.json`), which differs from `"1"` only in
  admitting `"increasing"` for `Fit.direction` and `Threshold.direction` and
  `"baseline_multiple"` for `Definition.kind`. `schema/scorecard-v1.json` is unchanged and
  `load_schema("1")` still loads it. `from_dict` reads both versions, and rejects a result tagged
  `"1"` that carries an increasing direction or a `baseline_multiple` definition. The check is per
  object: it looks at each `Fit` or `Threshold` and its own `schema_version`, not at the card as a
  whole.
- `robustmargin.testing.fake_fit` and `fake_threshold` take a keyword-only `direction`
  (`"decreasing"` by default, so existing calls are unchanged); `"increasing"` builds a schema-valid
  increasing-outcome result.
- `ratio_interval` accepts two thresholds of different directions on the same axis and definition;
  the ratio of severities does not depend on which way each outcome runs.

### Changed
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

### Fixed
- **Direct-likelihood fits could stop at a local maximum.** In 0.2.x this affects
  `fit_dose_response` with an estimated asymptote, or with a fixed one other than `upper=1`,
  `lower=0`. The optimiser ran from a single start, so it could return `OK` at a local maximum
  of the likelihood. In one reported design, a curve with log-likelihood −16.59 was reported
  `OK` while the maximum was −6.56. Every threshold, interval and ratio built on such a fit was
  wrong.
  - It could also report `NOT_CONVERGED` for data that do have a maximum.
  - The optimiser now also starts from five fixed points on the centred, scaled severity axis.
    It keeps a better start only when that start beats the original one's log-likelihood, and
    every convergence check then runs on that start. The method is deterministic.
  - A fit whose log-likelihood does not exceed that of the best constant curve (`s → ∞`) is now
    `NOT_CONVERGED`, with the wrong-sign/flat reason. On a log axis a zero-severity control stays
    at `upper` on that curve.
  - Migration: a direct-likelihood fit may now report a higher log-likelihood with different
    parameters, turn from `NOT_CONVERGED` to `OK`, or turn from `OK` to `NOT_CONVERGED` when its
    supremum lies on a boundary. A `NOT_CONVERGED` fit's first warning may name a different
    reason. `diagnose`'s per-link refits use the same path, so their AIC and status can change
    too. Fits with both asymptotes fixed at `1` and `0` are unaffected, since that likelihood has
    a single maximum. The exception is a direct-likelihood refit that does not beat the
    constant-curve log-likelihood, which is now `NOT_CONVERGED`.

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
