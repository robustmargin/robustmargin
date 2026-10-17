# R `drc` reference fixtures

`generate.R` fits the reference models in R's `drc` package (plus a few `glm`/`MASS` checks)
and writes the results as JSON under this directory. **Tests read only the JSON.** Running the
test suite never needs R installed (R is a local tool only, never an importable dependency of
robustmargin itself).

## Regenerating the fixtures

There is no host R install for this project. Run `generate.R` inside the pinned
`rocker/r-ver:4.6.1` container, with R packages installed into a local library directory (not
the image itself, so the image pull stays small and the library persists across runs):

```sh
docker run --rm -v "$PWD:/work" -v "$PWD/.claude/scratch/rlib:/rlib" -w /work rocker/r-ver:4.6.1 \
  Rscript -e '.libPaths(c("/rlib", .libPaths())); source("tools/drc_reference/generate.R")'
```

The library directory above is scratch (gitignored) and was populated once from the pinned
Posit Package Manager CRAN snapshot for `rocker/r-ver:4.6.1`:

```
options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/noble/2026-09-01"))
install.packages(c("drc", "MASS", "jsonlite"))
```

`MASS` ships with base R; the snapshot date only matters for `drc` and `jsonlite` (and their
transitive dependencies). If the scratch library is gone, reinstall from that same snapshot URL
so the package versions match what is recorded in every fixture's `versions` block, then re-run
the `docker run` command above. Every generated JSON file embeds `r_version`, `drc_version`,
`mass_version` and `jsonlite_version`, so a version drift shows up in the fixture itself, not
just in this file.

Versions used to generate the fixtures currently committed: R 4.6.1 (2026-06-24), drc 3.0.1,
MASS 7.3.65, jsonlite 2.0.0.

`write_json(..., digits = NA)` is used throughout, so every number is written at full `double`
precision rather than jsonlite's default rounding -- the `1e-4` (parameters) and `1e-3`
(interval endpoints) relative tolerances are applied by a *test* when comparing to robustmargin's
own output; the fixture itself should never be the source of the imprecision.

## Files this script writes

| File | Contents | Consumed by |
|---|---|---|
| `finney71.json` | `LN.2`/`LL.2`, `glm` probit/logit/cloglog, MASS `dose.p`, `ED` delta intervals | `tests/reference/test_models_drc.py`, `test_diagnostics_drc.py`, `test_threshold_drc_ed.py` |
| `finney71_profile.json` | the offset-trick profile-likelihood reference at the ED50, independent of drc | `tests/reference/test_threshold_profile.py` |
| `selenium_edcomp.json` | `EDcomp` with Fieller and delta intervals | `tests/reference/test_ratio_drc_edcomp.py` |
| `estimated_asymptotes.json` | a seeded simulated binomial dataset (embedded raw data) fit with binomial `LN.3`/`LN.4` | `tests/reference/test_models_drc.py` |
| `monotonicity_system_c_axis_1.json` | the one-sided Fisher exact p-value for the reversal between the two lowest levels of `tests/data/pooled_counts.csv`'s `system_c_axis_1` series | `tests/reference/test_monotonicity_drc.py` |

## Definitional notes (written here and in the fixtures themselves, never absorbed into a test tolerance)

**drc's `logLik()` is the full binomial log-likelihood, including the binomial coefficient.**
Confirmed in `finney71.json`'s `loglik_parity` block by recomputing the log-likelihood by hand
from drc's own fitted probabilities (with `lchoose`) and matching drc's `logLik()` value. This
also means dropping an all-success zero-severity control row from a GLM's likelihood (as
robustmargin's fixed `upper=1, lower=0` path does) is an exact no-op on the
log-likelihood: `finney71.json` shows the `glm` probit log-likelihood on the dose>0 subset
matching drc's `LN.2` log-likelihood on the full 6-row dataset (including the dose=0 control)
to the precision both optimizers converge to. The reason is a mathematical identity, not a
coincidence of this dataset: a row with zero failures contributes `lchoose(n, 0) + n*log(1) =
0` to the log-likelihood regardless of the fitted probability.

**`mu = log(e)` always, but the sign of `s` relative to drc's `b` is not universal — it depends
on which response was fit (failure vs. success) and which direction that response moves.** Each
fixture's `mapping` block records the sign actually found, confirmed empirically against
`predict()` (not merely assumed from a general rule):

- `finney71.json`: fit on `affected/total` (failure, **increasing** with dose). `LN.2` (probit):
  `s = 1/b` (`b` comes out positive, about `+1.83`). `LL.2` (logit): `s = -1/b` (`b` comes out
  negative, about `-3.10`). Both `LN.2` and `LL.2` fit the *same* underlying curve on the same
  data, so this is not a modelling choice -- it is what each `fct`'s internal parametrisation
  happens to produce, and it differs between them.
- `estimated_asymptotes.json`: fit on `successes/trials` directly (**decreasing** with
  severity, matching robustmargin's own `upper`/`lower` as success-rate asymptotes with no
  `1 - x` flip needed: `lower = c`, `upper = d`). For both `LN.3` and `LN.4`, `s = -1/b` (`b`
  comes out negative). The `mapping`
  block's `formula_check` note records that `P_success(x) = u - (u-l)*Phi((log(x)-mu)/s)` with
  `u = d`, `l = c` substituted in was checked against `predict()` at five held-out severities
  and matched exactly.

A reader deriving `mu`/`s` from a new drc fit not covered here should not assume either sign
without checking it against `predict()` the same way, and should record what was found rather
than reusing one of the two rules above by default.

**`MASS::dose.p`'s reported "Dose" is on the covariate's own scale, not the original dose
scale.** Because the `glm` formula used `log(dose)` as the covariate, `dose.p`'s output is
`mu` directly (`== -b0/b1`, the location on the *log*-dose axis), not `exp(mu)`. `dose.p`
applies no inverse transform for a transformed covariate; it does not know the covariate was
`log(dose)` rather than `dose`. `finney71.json`'s `mass_dose_p` block states this explicitly so
a test compares it to `mu`, never to `exp(mu)` or to drc's `e`.

**`estimated_asymptotes.json`'s `LN.3` fit is deliberately model-misspecified relative to the
simulation truth.** drc's `LN.3` fixes the lower limit `c` at exactly `0`, while the simulated
data's true lower asymptote is `0.1`. This is intentional: `LN.3` is the numerical-parity
reference for robustmargin's `upper="estimate", lower=0.0` call (a caller-chosen fixed lower
asymptote that happens not to match the truth), not a recovery-of-truth check. The
recovery-of-truth check lives in `tests/unit/test_models_recovery.py`, in Python, with both
asymptotes estimated (matching drc's `LN.4`, fixture also included here).

**Both binomial `LN.3` and `LN.4` fit successfully on the simulated data, including its exact
`severity = 0` control row.** No epsilon substitution, no dropped row, no convergence failure
was needed -- see `estimated_asymptotes.json`'s `feasibility_note`. There is nothing to report
as an infeasibility finding for this fixture.

**`LN.3` and `LN.4`'s *default* fits are under-converged, so each carries a second, `"tight"`
refit, and only the tight one is a reference value.** `drm()`'s default control
(`drmc()`: `relTol=1e-7`, `maxIt=500`) stops short of the actual maximum on this dataset:
robustmargin's `GenericLikelihoodModel` path (`bfgs` then a `newton` polish, run to a much tighter
tolerance) reaches a strictly *higher* log-likelihood than drc's default fit. This is not
robustmargin outperforming drc's method -- it is drc's default stopping early. Refitting from the
default fit's own coefficients (`start = coef(<default fit>)`) with
`control = drmc(relTol = 1e-14, maxIt = 10000, method = "BFGS")` closes almost all of that gap
(see the `LN.3`/`LN.4 logLik: default=... tight=...` lines `generate.R` prints when it runs).
Both fits are recorded under `drc.LN3`/`drc.LN4` as sibling `"default"` and `"tight"` blocks,
each with its own `coef`/`vcov`/`logLik` and a `control` string naming exactly what was passed
to `drm()`; `"default"` is kept only for the record, and **`tests/reference/test_models_drc.py`
reads only `"tight"`.**

Because even the tight refit has not converged to the usual 1e-4 relative precision (it has
converged to roughly 1e-3, judging by how far it moved from the default fit),
`TestEstimatedAsymptoteParity` does not use the ordinary bounds for this one class:
log-likelihood is checked as `robustmargin_logLik >= tight_logLik - 1e-6` (an inequality: robustmargin
is expected to do at least as well, not exactly as well, since it is the more tightly converged
of the two), `mu` uses an **absolute** tolerance of `1e-4` (its true simulated value is exactly
`0`, where a relative tolerance is ill-posed), and `s`/`upper`/`lower` use a relative `1e-3`
rather than `1e-4`. No other test in this file is affected -- the fixed-asymptote GLM tests
(`TestFinney71Probit`, `TestFinney71Logit`) still use the ordinary 1e-4 relative bound, because
that path has an exact, fully-converged `glm`/drc reference to compare against.
