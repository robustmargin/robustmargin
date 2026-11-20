# Changelog

All notable changes to robustmargin are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html). While the version is below 1.0, a
minor release may change the public API, and every such change carries a migration note.

## [Unreleased]

### Added
- **Design planning: `robustmargin.design`.** A submodule (not re-exported from the package root,
  like `report` and `testing`) that sizes a sweep from a pilot fit. `severity_grid` builds a
  geometric (log axis) or arithmetic (linear axis) grid with an optional zero-severity control.
  `plan_threshold_width` and `plan_ratio_width` take a binomial `Status.OK` `Fit` (two for a ratio)
  as the truth, simulate whole experiments at a candidate design, refit the pilot's specification
  (link, and the fixed or estimated asymptotes) and call the real `threshold` / `ratio_interval`,
  returning a `WidthPlan` with one width per replicate.
  - Clustered and paired designs (`n_clusters`, `icc`) use a Gaussian copula with a latent
    threshold, so each level's marginal success rate is the pilot curve for every `icc`. `icc` is
    the latent correlation, not the correlation of the binary outcomes (a latent 0.3 is about 0.19
    at a success rate of 0.5 and 0.10 at 0.95); the docstring gives the conversion. In a paired
    ratio the cross-arm latent correlation equals `icc`.
  - Width is `log(hi / lo)` (log-axis thresholds and all ratios) or `hi - lo` (linear-axis
    thresholds). A replicate with no finite width is `None` with a reason label in
    `WidthPlan.outcomes`, and counts as an infinite width in `quantile`; `share_at_most` gives the
    share of replicates at or under a width with its binomial plug-in standard error, which is 0
    at a share of 0 or 1.
  - Outcome labels are `"OK"`, a non-`OK` `Status` value, a non-`NONE` `Censoring` value, a
    non-`BOUNDED` `IntervalShape` value, `"BOTH_SIDES_OPEN"` (a threshold with no finite end on
    either side) and `"LOWER_AT_ZERO"` (a bounded ratio whose lower end is not positive).
    `WidthPlan` validates every field on construction.
  - `n_resamples` sets the bootstrap resample count for a bootstrap interval or ratio method (it is
    an error for any other) and is recorded on the `WidthPlan`. Bootstrap intervals draw their seed
    from the replicate's own stream, so a bootstrap design is reproducible for any `workers`.
  - Replicate streams are spawned from `numpy.random.SeedSequence(seed)`, so the result is identical
    for any `workers`; `workers > 1` uses `concurrent.futures.ProcessPoolExecutor`, which under the
    `spawn` start method needs the calling script to guard its entry point with
    `if __name__ == "__main__":`.
  - Gaussian and isotonic pilot fits raise `ValueError`.
  - `report.to_dict` / `report.from_dict` read and write a `WidthPlan` as its own document
    (`"type": "WidthPlan"`, `plan_schema_version` `"1"`). The score-card schema stays `"3"`. A
    `WidthPlan` inside a `Scorecard`'s `results` is rejected with `ValueError`.
- **Isotonic fits: `fit_dose_response(obs, model="isotonic")`.** A monotone, parameter-free fit to
  counts or to a continuous response. `link`, `upper` and `lower` must be omitted (`ValueError`
  if given). The fit is the weighted least-squares monotone fit to the cell rates (weights
  `trials`) or cell means (weights `n`) by pool-adjacent-violators, with `scipy.optimize.
  isotonic_regression`.
  - `Fit.cell_estimates` (new, last field, `None` for every other model) holds the fitted value of
    every cell, the control included. An isotonic `Fit` has `link="none"`, no `params`,
    `covariance`, `diagnostics` or `residual`, a `log_likelihood` (counts include `log C(n, k)`, as the parametric binomial fits do), and status
    `OK` always.
  - `Fit.predict` and `diagnose` raise `ValueError` for an isotonic fit, naming `cell_estimates`.
  - For `fit_dose_response` with `model="binomial"` or `"gaussian"`, a missing `link`, `upper` or
    `lower` still raises `TypeError`, now naming each missing argument; an explicit `None` raises
    what it raised before.
  - A continuous isotonic fit with a zero residual sum of squares (every cell mean already
    monotone and no cell with any spread) raises `ValueError`: its likelihood is unbounded.
- **Isotonic thresholds.** The value is where the line through `cell_estimates` over the tested
  severities (on `log` severity for a log axis, where the control is not a node) first reaches the
  target. Nothing is extrapolated: a curve that never reaches the target inside the tested levels,
  or has already reached it at the lowest, is `UNREACHABLE` with no number and a warning that names
  the side. For counts the exact `RIGHT`, `LEFT` and `FAILS_AT_BASELINE` tests run first, as for
  any fit; a continuous isotonic fit makes no exact claim. `relative` raises;
  `baseline_fraction` and `baseline_multiple` use the isotonic value at the control as the
  baseline and need one. `ratio_interval` raises `ValueError` for a threshold from an isotonic fit.
- **Bootstrap intervals.** `threshold` takes `interval_method="parametric_bootstrap"` (binomial and
  Gaussian fits: simulate from the fitted model, refit the same model) and `"cell_bootstrap"`
  (isotonic fits only, and binomial fits: resample each level from its own data), with
  `n_resamples` (default 999) and `seed`. An isotonic fit takes only `"cell_bootstrap"`, and a
  Gaussian fit does not take it. Passing `n_resamples` or `seed` with `"profile"` or `"delta"`
  raises `ValueError`.
  - The interval is the percentile interval of the replicate thresholds (the `"weibull"` quantile
    method of `numpy.quantile`). A replicate whose target is unreachable counts as `-inf` or `+inf`
    by the side it implies, and a side is open (`OPEN_LOWER`, `OPEN_UPPER`) when more than `alpha/2`
    of the replicates are unreachable on it; both open gives no bounds and a warning.
  - A replicate whose refit fails, or whose baseline is undefined (at or below zero), is counted
    as failed. It enters the percentiles as `-inf` for the lower end and `+inf` for the upper end,
    so a failure can only widen the interval, and it counts toward an open side. Above 1% a
    `BOOTSTRAP_FAILURES:` warning states `failed/n_resamples`; above `min(10%, alpha/2)` the
    result is `NOT_CONVERGED` with `interval_method` `"exact_bound"` and the exact-bound fallback for counts
    (no numbers for a continuous response).
  - `BOOTSTRAP_FEW_RESAMPLES:` warns when `n_resamples < 2/alpha - 1`. An interval of zero width is
    reported with no bounds and a `BOOTSTRAP_ZERO_WIDTH:` warning.
  - A percentile interval that excludes its own point estimate is reported as computed, with a
    `BOOTSTRAP_EXCLUDES_ESTIMATE:` warning that states `z0`, and a `Threshold` that does so
    without the warning is rejected. For the bootstrap methods the `Threshold` invariant is
    `lo <= hi`; `"profile"` and `"delta"` keep `lo <= value <= hi`.
  - A counts `cell_bootstrap` draws `Binomial(n, (k + 0.5)/(n + 1))` for each level, so a level
    observed at 0 or `n` is not resampled as certain. A continuous `cell_bootstrap` draws each cell
    mean from `N(mean, sd^2 (n - 1) / n^2)` (a mean under resampling with replacement), since a
    cell keeps no raw responses; a cell of one observation raises `ValueError`.
  - A bootstrap of a fit with an estimated asymptote refits from the starts found once for the
    fitted data, not from a scan per replicate, and is still far slower than one with fixed
    asymptotes.
- **`BootstrapSummary`** (exported): `n_resamples`, `failed`, `unreachable_low`,
  `unreachable_high` and `seed`. `Threshold.bootstrap` (new, last field, default `None`) holds it
  exactly when a bootstrap ran. With `seed=None` a seed is drawn and recorded, so every bootstrap
  result can be reproduced bit for bit.
- `Threshold` accepts `NOT_CONVERGED` on an `OK` fit exactly when `bootstrap` is set (a bootstrap
  that gave up).
- `robustmargin.testing`: `fake_fit(model="isotonic")`, `fake_threshold(model="isotonic")` and
  `fake_threshold(interval_method=)`, which takes `"profile"`, `"delta"`,
  `"parametric_bootstrap"` and `"cell_bootstrap"`.
- **Clustered fits: `fit_dose_response(obs, ..., dependence="clustered")`.** For binomial fits of
  `Observations` that carry `cluster` ids. The point estimate, status and log-likelihood are those
  of the independent fit (a working-independence maximum likelihood); the covariance is the
  cluster-robust (CR1) sandwich, `G/(G - 1)` times the sum over clusters of the outer product of
  each cluster's summed score, around the fit's observed-information inverse. Gaussian and isotonic
  fits raise `ValueError` ("clustered continuous inference is not implemented"), and so does
  missing `cluster`. A fit with no more clusters than estimated parameters raises `ValueError`
  before fitting ("too few clusters for a cluster-robust covariance").
  - `Fit.dependence` (`"independent"` or `"clustered"`) and `Fit.cluster_cells` (new, last fields)
    say which covariance a fit carries. `cluster_cells` holds one `ClusterCell` (`severity`,
    `cluster`, `successes`, `trials`) per severity and cluster, and is set exactly when
    `dependence == "clustered"`. `Observations.cluster_cells()` returns the same counts, or `None`
    when unclustered. `ClusterCell` and `ClusterAdjacentPair` are exported.
  - A clustered fit whose sandwich is not positive definite is `NOT_CONVERGED` with a
    `CLUSTER_COVARIANCE:` warning and no parameters, even where the independent fit converged.
  - Below 20 clusters a clustered fit carries `FEW_CLUSTERS:`, because its `predict` band rests on
    the sandwich.
  - `Diagnostics.warnings` carries `CLUSTERED_GOF:`: the goodness-of-fit test is a test of the
    independent-binomial model and is expected to reject under clustering, and each link's AIC uses
    the working-independence likelihood, so it overstates the evidence by about the design effect.
  - `Fit.covariance`, `Fit.predict` and `"delta"` use the sandwich on a clustered fit; read
    `Fit.dependence` to know which.
- **Clustered thresholds.** `threshold(fit, ..., dependence="clustered")` requires a clustered fit
  (and a clustered fit requires `"clustered"`); a mismatch raises `ValueError`. `interval_method`
  is `"delta"` (the recommended method: the sandwich with a Student-t quantile on `G - 1`
  degrees of freedom) or `"cluster_bootstrap"` (an alternative that needs about 40 or more
  clusters: draw `G` clusters with replacement, refit, and take percentiles; the replicate
  counting, failures, open sides and warnings of the other bootstraps apply, and
  `Threshold.bootstrap` is set). `"profile"`, `"parametric_bootstrap"` and `"cell_bootstrap"` raise
  `ValueError`, and so does `"cluster_bootstrap"` on a fit that is not clustered.
  - A clustered threshold makes no exact claim: it is never `FAILS_AT_BASELINE`, `RIGHT` or
    `LEFT`. A crossing beyond the tested range is `EXTRAPOLATED` with an interval, and a failed fit
    keeps its status with `interval_method="exact_bound"` and no value.
  - The sandwich standard error is computed beside the bootstrap. When the bootstrap spread (scaled
    by `sqrt(G / (G - 1))`) or location departs from it by more than a stated tolerance, the result
    carries a `CLUSTER_CROSS_CHECK:` warning. The numbers never change. It is skipped when either
    interval is not closed.
  - Few clusters: a clustered threshold carries `FEW_CLUSTERS:` below 20 clusters with `"delta"`,
    and below 40 with `"cluster_bootstrap"`, whose percentile interval under-covers there; that
    warning names `"delta"`.
  - With unbalanced clusters the estimand is the trial-weighted pooled curve.
- **Cluster sign test: `check_monotonicity(obs, dependence="clustered")`.** For each adjacent pair
  of severities, an exact one-sided sign test over the clusters observed at both levels: with `r`
  clusters whose rate moves against `direction` and `o` that move with it, `p = P(Binomial(r + o,
  1/2) >= r)`, Holm-adjusted across all pairs. Ties are dropped, and each cluster counts once
  whatever its size. The null is that every cluster's own curve is monotone, so this is a
  median-direction test, not a test of the pooled curve's monotonicity. Only balanced clusters (equal trials at both levels) take part; the rest are
  counted in `ClusterAdjacentPair.clusters_unbalanced` and a warning says so. Further warnings say
  when no cluster is observed at both levels of a pair and when the test cannot flag anything at
  `alpha`. The result has `dependence="clustered"` and
  `method="cluster_sign_test_one_sided_holm"`. It requires `cluster` and count `Observations`
  (`ContinuousObservations` raises `ValueError`).
- **The paired ratio: `ratio_interval(t_a, t_b, method=..., dependence="paired")`.** For two
  thresholds on the same clusters. `method` is `"cluster_bootstrap"` (new; each replicate draws one
  set of cluster ids and refits both fits, so the correlation is kept), `"fieller"` or
  `"log_delta"`. The last two use the cross-covariance of the two fits' cluster-robust estimates,
  recomputed from the stored cluster counts and parameters, with Student-t quantiles on `G - 1`
  degrees of freedom. `"paired"` requires both thresholds clustered, on fits with the same cluster
  ids and the usual matching axis, unit and definition (`ValueError` otherwise);
  `"cluster_bootstrap"` requires `"paired"`; `n_resamples` and `seed` are accepted only with
  `"cluster_bootstrap"`.
  - Under `"paired"` the `B <= 0` rule can hold, so `EXCLUSIVE` is now reachable, and a Fieller set
    that is bounded above only is reported as `BOUNDED` `[0, K / (2B)]`.
  - `ratio_interval(..., dependence="independent")` on clustered thresholds (disjoint cluster sets,
    or one clustered and one not) uses a Student-t quantile with Welch-Satterthwaite degrees of
    freedom built from `(V_a, G_a - 1)` and `(V_b, G_b - 1)`, as Gaussian thresholds already did; an
    unclustered binomial input contributes infinite degrees of freedom.
  - `Ratio.bootstrap` (new, last field, a `BootstrapSummary`) is set when the cluster bootstrap
    ran. A replicate with one unreachable threshold is a ratio of 0 or infinity and counts toward
    an open side; one with both unreachable on the same side is undefined and counts as failed. A
    `RATIO_BOOTSTRAP_CENSORED:` warning gives the share of replicates with a censored threshold and
    the number undefined, when that share is large. Give-up, open sides and `BOOTSTRAP_*` warnings
    follow the threshold bootstrap, and `CLUSTER_CROSS_CHECK:` compares with the paired
    `"log_delta"` (log axis) or `"fieller"` (linear axis). `FEW_CLUSTERS:` is carried from the
    inputs, below 20 clusters for `"fieller"` and `"log_delta"` and below 40 for
    `"cluster_bootstrap"`. Failure causes are counted per replicate, so a replicate in which both
    fits fail for the same cause counts once. The percentile interval need not contain the estimate.
- **Testing fakes.** `fake_fit`, `fake_threshold` and `fake_ratio` take `dependence=`.
  `fake_fit(dependence="clustered")` builds a schema-valid clustered binomial fit;
  `fake_threshold` follows it (default `"cluster_bootstrap"`); `fake_ratio(dependence="paired")`
  builds two clustered fits on shared clusters (default `"cluster_bootstrap"`, `BOUNDED`).
- **Schema `"3"` additions.** `Fit.dependence` and `Fit.cluster_cells`, `Threshold.dependence`
  `"clustered"` and `interval_method` `"cluster_bootstrap"`, `Ratio.dependence` `"paired"`,
  `Ratio.method` `"cluster_bootstrap"` and `Ratio.bootstrap`, `MonotonicityCheck.dependence`
  `"clustered"` with the sign-test method and its pair type. A card tagged `"1"` or `"2"` that
  carries any of them is rejected; on such a card `to_dict` leaves out `Fit.dependence` when it is
  `"independent"`.

### Changed
- **A paired ratio no longer raises when its inputs are clustered fits on shared clusters.**
  `ratio_interval(..., dependence="paired")` used to raise `ValueError`; it now returns a `Ratio`
  (see above), and `"cluster_bootstrap"` is a new `method`. This changes a raise into a result.
  - **Migration.** No existing call changes behaviour: `"paired"` and `"clustered"` are reachable
    only with the new inputs (clustered fits and thresholds). A paired ratio of two thresholds from
    one fit still raises, as does `"independent"` on clustered thresholds that share a cluster.
    Code that relied on the `ValueError` as a guard should check `Ratio.dependence` instead.
- **`Fit.predict` on a clustered fit uses a Student-t quantile on `G - 1` degrees of freedom** for
  its confidence band, like every other clustered sandwich interval. An independent fit is
  unchanged.
- **Score-card schema `"3"`** (`schema/scorecard-v3.json`). It is `"2"` plus: `Fit.model`
  `"isotonic"`, `Fit.link` `"none"`, `Threshold.interval_method` `"parametric_bootstrap"` and
  `"cell_bootstrap"`, a nullable `Fit.cell_estimates` and a nullable `Threshold.bootstrap`.
  `schema/scorecard-v1.json` and `schema/scorecard-v2.json` are unchanged.
  - **Migration.** Every result now has `schema_version` `"3"` by default. On a `"3"` result
    `to_dict` writes `cell_estimates` and `bootstrap`, as `null` when unset. `from_dict` reads
    `"1"`, `"2"` and `"3"`; a result read from an older card keeps its `schema_version`, and
    `to_dict` writes it back without the two new keys when they are unset, so the release that
    wrote the card can still read it. A payload tagged `"1"` or `"2"` that carries a non-null
    `cell_estimates` or `bootstrap`, the isotonic model, the `"none"` link or a bootstrap interval
    method is rejected. A reader
    older than this release rejects every card written now (the keys are unknown to it), so upgrade
    readers before writers.
  - Code that pins `schema_version == "2"` for a result it builds (for example a test of a
    serialised card) changes to `"3"`. Both new fields are last and default to `None`, so existing
    keyword construction of `Fit` and `Threshold` is unaffected (the result types are keyword-only).
- **SciPy floor `>=1.12`** (was `>=1.11.1`): `scipy.optimize.isotonic_regression` arrived in 1.12.
- The package metadata names its author, Babak Badkoubeh, in place of "robustmargin
  contributors", and links the Zenodo DOI and the author's ORCID. A `NOTICE` file carries the
  copyright line and ships in the sdist and wheel beside `LICENSE`. The README gains a "Citing"
  section.

### Fixed
- **`Fit.predict` confidence band for a binomial fit with both asymptotes fixed.** The delta-method
  variance of `z = (t - mu)/s` had the wrong sign on the `mu`-`s` covariance term: the term is
  `+2z Cov(mu, s)`, not `-2z Cov(mu, s)`. The band was too wide on one side of the curve's midpoint
  and too narrow on the other. Present in every release up to 0.3.0. The affected outputs are
  `lo` and `hi` of `Fit.predict` for every binomial fit with both asymptotes fixed, in both
  directions and with every link; the estimates and every threshold are unchanged. Bands for fits
  with an estimated asymptote were unaffected. Recompute any stored bands after upgrading.
- **Profile interval with an estimated asymptote: finite upper bound too narrow at the
  reachability wall.** When the constrained maximum lay at the reachability wall far beyond the
  tested range, the profile could report a finite upper bound that was too narrow. The profile
  now also uses, at each candidate severity, the log-likelihood of a feasible curve through the
  target as a lower bound on the constrained maximum, so a bound cannot move inward. This
  extends to `baseline_fraction` and `baseline_multiple` targets. Where the interval truly has no
  finite upper end it is now reported open, as it was for `absolute` targets. Fits with both
  asymptotes fixed are unchanged.

### Known limitations
- Profile bounds with an estimated asymptote can still be too narrow when the constrained maximum
  lies in an inner basin narrower than the scan grid used to place the optimiser's starts.
- `Fit.predict` on a count fit with an estimated asymptote: at a severity far beyond the tested
  range, where the fitted curve equals its asymptote in double precision, `lo` and `hi` are `nan`;
  slightly closer in, they can collapse onto the estimate. The estimate itself is unaffected.
  Read the band within or near the tested range.
- For `baseline_fraction` and `baseline_multiple` targets with an estimated upper asymptote, the
  far-extrapolation bound uses the curve at the estimated upper asymptote, so an upper bound can
  still be slightly narrower than the full profile allows. It is never narrower than before.

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
