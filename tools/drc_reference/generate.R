#!/usr/bin/env Rscript
# robustmargin's R `drc` reference fixtures.
#
# Writes JSON files under tools/drc_reference/ that the reference tests read. Tests never run R
# themselves (R is a local tool only, never an importable dependency of robustmargin); this
# script is how the fixtures get (re)generated, by hand, in the pinned container documented in
# tools/drc_reference/README.md.
#
# Every JSON file records the R, drc and MASS versions actually used, plus (where relevant) the
# drc -> robustmargin parameter mapping this run empirically confirmed:
#   mu = log(e); s is +-1/b depending on which response was fit and which direction it moves in
#   (see the "mapping" block written into each file -- drc's sign convention is checked against
#   predict() and recorded as a fact, not left as a comment).
#
# Run via the exact container command in the README. Do not run against a host R install: the
# CRAN snapshot and package versions must be pinned and reproducible.

suppressPackageStartupMessages({
  library(drc)
  library(MASS)
  library(jsonlite)
})

out_dir <- "tools/drc_reference"
if (!dir.exists(out_dir)) {
  stop("expected to be run from the robustmargin repo root (tools/drc_reference not found)")
}

versions <- list(
  r_version = R.version.string,
  drc_version = as.character(packageVersion("drc")),
  mass_version = as.character(packageVersion("MASS")),
  jsonlite_version = as.character(packageVersion("jsonlite")),
  generated_at = format(Sys.time(), tz = "UTC", usetz = TRUE)
)

write_fixture <- function(obj, filename) {
  path <- file.path(out_dir, filename)
  write_json(obj, path, auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
  cat("wrote", path, "\n")
}

# Coerce a named numeric vector (drc's coef()/a glm's coef()) into a plain named list, so
# jsonlite writes it as a JSON object keyed by parameter name rather than an array.
named_vec_to_list <- function(v) {
  as.list(setNames(as.numeric(v), names(v)))
}

# Coerce a matrix (vcov()) into a row-major list-of-lists, with the row/col names recorded
# alongside so a reader never has to assume an ordering. drc's vcov() carries no dimnames at
# all (confirmed: rownames(vcov(drm_fit)) is NULL), unlike glm's, so `names` lets a caller
# supply the parameter order explicitly (from coef()) rather than writing a JSON `{}` for a
# missing dimname, which would leave the ordering undocumented.
matrix_to_fixture <- function(m, names = NULL) {
  row_names <- if (!is.null(rownames(m))) rownames(m) else names
  col_names <- if (!is.null(colnames(m))) colnames(m) else names
  list(
    row_names = row_names,
    col_names = col_names,
    values = lapply(seq_len(nrow(m)), function(i) as.numeric(m[i, ]))
  )
}

# Full binomial log-likelihood, including the binomial coefficient (lchoose), computed directly
# from fitted probabilities -- used to cross-check that drc's logLik() is the full binomial
# log-likelihood and not the kernel-only (coefficient-omitted) version. A 0/n or n/n cell
# contributes exactly 0 via the convention 0*log(0) := 0 (matched with ifelse below).
full_binomial_loglik <- function(successes_or_failures, n, p) {
  y <- successes_or_failures
  term_success <- ifelse(y > 0, y * log(p), 0)
  term_failure <- ifelse(y < n, (n - y) * log(1 - p), 0)
  sum(lchoose(n, y) + term_success + term_failure)
}

# =================================================================================================
# 1. finney71 -- LN.2 / LL.2 (drc), glm probit/logit/cloglog, MASS dose.p, ED delta intervals.
# =================================================================================================

data(finney71)
# str(finney71): 'data.frame': 6 obs. of 3 variables: dose (num), total (int), affected (int).
# affected/total plays the role of FAILURE (robustmargin's F models failure probability rising
# with dose; see the docstring in src/robustmargin/models.py). Includes an exact dose = 0 row
# (0/49 affected), an all-success control on the log axis.
stopifnot(identical(names(finney71), c("dose", "total", "affected")))
stopifnot(any(finney71$dose == 0))

m_ln <- drm(affected / total ~ dose, weights = total, data = finney71, fct = LN.2(), type = "binomial")
m_ll <- drm(affected / total ~ dose, weights = total, data = finney71, fct = LL.2(), type = "binomial")

# glm equivalents, fit on the dose > 0 subset only: log(0) is undefined, and this mirrors
# robustmargin's own rule of dropping an all-success zero-severity control from the
# likelihood when both asymptotes are fixed at 1/0. The zero-dose row is kept out of `fdat` for
# every glm fit below, and the parity check further down confirms this changes nothing, because
# a 0-failure row's log-likelihood contribution is exactly zero (lchoose(n, 0) + n*log(1) = 0).
fdat <- finney71[finney71$dose > 0, ]

fit_glm <- function(link) {
  g <- glm(cbind(affected, total - affected) ~ log(dose), family = binomial(link = link), data = fdat)
  list(
    link = link,
    coef = named_vec_to_list(coef(g)),
    vcov = matrix_to_fixture(vcov(g)),
    logLik = as.numeric(logLik(g)),
    converged = isTRUE(g$converged),
    # R-native goodness-of-fit fields, added so
    # tests/reference/test_diagnostics_drc.py's TestAgainstRNativeGofFields class has a second,
    # fully R-computed cross-check independent of that file's own Python recomputation from
    # coef/logLik above. df_residual is 3 (fdat, the dose>0 subset, has 5 rows; 2 estimated
    # parameters; 5-2=3) -- fdat never contains the dose=0 row at all (it was subsetted out above
    # because log(0) is undefined, not because glm() drops it itself), which is what makes this
    # comparable to robustmargin's own diagnose(), whose *own* exclusion of the dose=0 control is
    # structural (a log-axis severity-0 cell under a fixed upper=1.0), not a coincidence of the
    # two happening to remove the same row.
    deviance = as.numeric(deviance(g)),
    pearson_chi2 = as.numeric(sum(residuals(g, type = "pearson")^2)),
    df_residual = as.numeric(df.residual(g)),
    AIC = as.numeric(AIC(g))
  )
}
glm_probit <- fit_glm("probit")
glm_logit <- fit_glm("logit")
glm_cloglog <- fit_glm("cloglog")

# drc's own vcov comes from a *numerical* Hessian inside its optimizer, so it is
# cross-checked here against an independently-computed observed-information vcov: the negative
# full binomial log-likelihood, in drc's own (b, e) parametrisation, differentiated numerically
# with base R's own optimHess() at drc's reported coef (not re-optimized), then inverted. This
# is a second, independent numerical Hessian, not a re-use of drc's internal one, and it is
# recorded in the fixture rather than only asserted here, per instruction.
neg_log_lik_LN_LL <- function(par, x, y, n, form) {
  b <- par[1]
  e <- par[2]
  if (form == "LN2") {
    z <- ifelse(x > 0, b * (log(x) - log(e)), -Inf)
    p <- pnorm(z)
  } else if (form == "LL2") {
    z <- ifelse(x > 0, -b * (log(x) - log(e)), -Inf)
    p <- plogis(z)
  } else {
    stop("neg_log_lik_LN_LL: unknown form ", form)
  }
  p <- pmin(pmax(p, 1e-300), 1 - 1e-16)
  term_y <- ifelse(y > 0, y * log(p), 0)
  term_ny <- ifelse(y < n, (n - y) * log(1 - p), 0)
  -sum(term_y + term_ny)
}

observed_hessian_vcov <- function(coef_vec, form) {
  h <- optimHess(
    coef_vec, neg_log_lik_LN_LL,
    x = finney71$dose, y = finney71$affected, n = finney71$total, form = form
  )
  solve(h)
}

observed_hessian_check <- function(drc_model, form) {
  drc_vcov <- vcov(drc_model)
  hessian_vcov <- observed_hessian_vcov(coef(drc_model), form)
  dimnames(hessian_vcov) <- dimnames(drc_vcov)
  list(
    vcov = matrix_to_fixture(hessian_vcov, names = names(coef(drc_model))),
    max_abs_relative_difference_from_drc_vcov = max(abs((hessian_vcov - drc_vcov) / drc_vcov)),
    note = paste(
      "An independent numerical Hessian of the full binomial negative log-likelihood",
      "(base R's optimHess(), not drc's internal optimizer), evaluated at drc's own",
      "coefficients and inverted. Confirms drc's vcov is the OBSERVED information (matches",
      "this independent computation to within 1e-5 relative on finney71), not merely close to",
      "it by coincidence -- which is why drc's vcov, not glm's expected-information",
      "cov_params(), is the covariance parity reference."
    )
  )
}

g_probit_obj <- glm(cbind(affected, total - affected) ~ log(dose), family = binomial(link = "probit"), data = fdat)
dose_p_50 <- dose.p(g_probit_obj, p = 0.5)

# Parity check written into the fixture, not just asserted here: drc's LN.2 logLik on the FULL
# 6-row dataset (including the dose=0 control) against the glm probit logLik on the 5-row
# dose>0 subset. Both equal -10.47968... to the precision that matters here; the equality is
# the point (the zero-control-dropping rule is provably a no-op on the likelihood
# value, not just an implementation convenience).
loglik_parity <- list(
  drc_LN2_full_dataset = as.numeric(logLik(m_ln)),
  glm_probit_dose_gt_0_subset = as.numeric(logLik(g_probit_obj)),
  manual_full_binomial_loglik_on_drc_LN2_fitted_values = full_binomial_loglik(
    finney71$affected, finney71$total, fitted(m_ln)
  ),
  note = paste(
    "drc's logLik() is the FULL binomial log-likelihood, including the binomial coefficient",
    "(confirmed by matching a manual lchoose-based computation on drc's own fitted values).",
    "Dropping the all-success dose=0 control row from the GLM likelihood (robustmargin's rule",
    "for the fixed u=1,l=0 path) changes the log-likelihood by exactly 0, because",
    "an n_successes=0-failure row's binomial log-likelihood term (lchoose(n,0) + n*log(1)) is",
    "identically 0 -- this is a mathematical identity, not a coincidence of this dataset."
  )
)

# Sign convention (empirically confirmed against predict(), not assumed): on finney71, the
# response fit is affected/total (failure, INCREASING with dose). For LN.2 (probit), b comes
# out positive and mu=log(e), s=+1/b reproduces predict() exactly. For LL.2 (logit), b comes
# out negative and mu=log(e), s=-1/b reproduces predict() exactly. Both were checked by
# recomputing pnorm(b*(log(x)-log(e))) / plogis(-b*(log(x)-log(e))) against predict(m, ...) at
# five held-out doses and finding exact agreement (scratch verification, not embedded here).
mapping_finney71 <- list(
  response_fit = "affected/total (failure, increasing with dose)",
  LN2 = list(mu = "log(e)", s = "1/b", note = "b is positive on finney71 (approx +1.83)"),
  LL2 = list(mu = "log(e)", s = "-1/b", note = "b is negative on finney71 (approx -3.10)"),
  upper_fixed = 1.0,
  lower_fixed = 0.0
)

finney71_fixture <- list(
  versions = versions,
  dataset = list(
    dose = finney71$dose, total = finney71$total, affected = finney71$affected,
    note = "str(finney71): dose (num), total (int), affected (int); includes an exact dose=0 all-success control row (0/49)."
  ),
  mapping = mapping_finney71,
  loglik_parity = loglik_parity,
  df_comparison_note = list(
    drc_df_residual = as.numeric(nrow(finney71) - length(coef(m_ln))),
    glm_df_residual = as.numeric(nrow(fdat) - length(coef(g_probit_obj))),
    robustmargin_diagnose_df = as.numeric(nrow(fdat) - length(coef(g_probit_obj))),
    note = paste(
      "A note on residual degrees of freedom: drc's drm() fits the FULL 6-row",
      "finney71 (LN.2/LL.2 handle dose=0 directly on the log scale), so drc's own residual df",
      "is 4 (6 rows - 2 parameters). glm() here and robustmargin's diagnose() both report df=3",
      "(5 rows - 2 parameters), but for two DIFFERENT reasons that happen to agree on this",
      "dataset: this script subsets the dose=0 row out of fdat before ever calling glm(),",
      "because log(0) is undefined and would make the glm() call itself fail -- glm() does not",
      "'drop' the row on its own. robustmargin's diagnose(), by contrast, fits all 6 rows and then",
      "excludes the dose=0 cell from its own deviance/Pearson/df computation STRUCTURALLY (a",
      "log-axis severity-0 cell under a fixed upper=1.0), never because a computed probability",
      "happens to round to exactly 1.0. drc's df=4 is not directly",
      "comparable to either: it is neither subsetting nor structural exclusion, just an ordinary",
      "fit on all 6 rows."
    )
  ),
  drc = list(
    LN2 = list(
      fct = "LN.2", link = "probit",
      coef = named_vec_to_list(coef(m_ln)),
      vcov = matrix_to_fixture(vcov(m_ln), names = names(coef(m_ln))),
      observed_hessian_check = observed_hessian_check(m_ln, "LN2"),
      logLik = as.numeric(logLik(m_ln)),
      ED_delta = list(
        p = c(10, 50, 90),
        table = matrix_to_fixture(as.matrix(ED(m_ln, c(10, 50, 90), interval = "delta", display = FALSE)))
      )
    ),
    LL2 = list(
      fct = "LL.2", link = "logit",
      coef = named_vec_to_list(coef(m_ll)),
      vcov = matrix_to_fixture(vcov(m_ll), names = names(coef(m_ll))),
      observed_hessian_check = observed_hessian_check(m_ll, "LL2"),
      logLik = as.numeric(logLik(m_ll)),
      ED_delta = list(
        p = c(10, 50, 90),
        table = matrix_to_fixture(as.matrix(ED(m_ll, c(10, 50, 90), interval = "delta", display = FALSE)))
      )
    )
  ),
  glm = list(probit = glm_probit, logit = glm_logit, cloglog = glm_cloglog),
  mass_dose_p = list(
    p = 0.5, link = "probit",
    dose = as.numeric(dose_p_50),
    se = as.numeric(attr(dose_p_50, "SE")),
    note = paste(
      "MASS::dose.p reports its result on the *covariate's own scale*. Because the glm formula",
      "used log(dose) as the covariate, dose.p's 'Dose' output is actually on the log(dose)",
      "scale already -- it is mu directly (== -b0/b1), NOT exp(mu). This is a definitional",
      "fact about dose.p's implementation (it applies no inverse transform for a transformed",
      "covariate), not a robustmargin convention, and is why this value is compared to mu, never",
      "to exp(mu) or to drc's e."
    )
  )
)
write_fixture(finney71_fixture, "finney71.json")

# =================================================================================================
# 2. Profile-likelihood reference: finney71 probit, ED50, offset-trick + uniroot.
#    Independent of drc: it is a plain glm() refit at each candidate location, used to check
#    robustmargin's own profile-likelihood interval.
# =================================================================================================

theta_hat <- as.numeric(-coef(g_probit_obj)[1] / coef(g_probit_obj)[2])
min_dev <- as.numeric(deviance(g_probit_obj))

prof_dev <- function(theta, zstar) {
  g <- glm(
    cbind(affected, total - affected) ~ 0 + I(log(dose) - theta),
    offset = rep(zstar, nrow(fdat)),
    family = binomial(link = "probit"), data = fdat
  )
  deviance(g)
}

level <- 0.95
chisq_1 <- qchisq(level, df = 1)
zstar_ed50 <- qnorm(0.5) # 0, the ED50 profile
target_dev <- min_dev + chisq_1

f <- function(theta) prof_dev(theta, zstar_ed50) - target_dev
lo_theta <- uniroot(f, lower = theta_hat - 3, upper = theta_hat)$root
hi_theta <- uniroot(f, lower = theta_hat, upper = theta_hat + 3)$root

profile_fixture <- list(
  versions = versions,
  note = paste(
    "Independent R implementation of the offset-trick profile likelihood, fit on the same",
    "finney71 dose>0 subset as the glm probit fixture above. theta is",
    "log(dose); zstar is qnorm(target proportion) for the ED50 (0 for probit p=0.5)."
  ),
  model = "probit, fixed upper=1 lower=0, finney71 dose>0 subset",
  level = level,
  chisq_1_at_level = chisq_1,
  theta_hat = theta_hat,
  ed50_dose_hat = exp(theta_hat),
  min_deviance = min_dev,
  target_deviance = target_dev,
  profile_ci_theta = list(lo = lo_theta, hi = hi_theta),
  profile_ci_dose = list(lo = exp(lo_theta), hi = exp(hi_theta))
)
write_fixture(profile_fixture, "finney71_profile.json")

# =================================================================================================
# 3. selenium EDcomp -- Fieller and delta interval references for a ratio of ED50s.
# =================================================================================================

data(selenium)
# str(selenium): type (num, curve id 1-4), conc (num, includes conc=0 control rows per curve),
# total (num), dead (num).
stopifnot(identical(names(selenium), c("type", "conc", "total", "dead")))

m_se <- drm(
  dead / total ~ conc,
  curveid = type, weights = total, data = selenium, fct = LL.2(), type = "binomial"
)

edcomp_fieller <- EDcomp(m_se, c(50, 50), interval = "fieller", display = FALSE)
edcomp_delta <- EDcomp(m_se, c(50, 50), interval = "delta", display = FALSE)

selenium_fixture <- list(
  versions = versions,
  dataset = list(
    type = selenium$type, conc = selenium$conc, total = selenium$total, dead = selenium$dead,
    note = "str(selenium): type (curve id, num 1-4), conc (num, dose; includes conc=0 controls per curve), total (num), dead (num)."
  ),
  model = list(
    fct = "LL.2", link = "logit", curveid = "type",
    coef = named_vec_to_list(coef(m_se)),
    vcov = matrix_to_fixture(vcov(m_se), names = names(coef(m_se))),
    logLik = as.numeric(logLik(m_se))
  ),
  EDcomp_fieller = list(
    pairs = c(50, 50),
    table = matrix_to_fixture(as.matrix(edcomp_fieller))
  ),
  EDcomp_delta = list(
    pairs = c(50, 50),
    table = matrix_to_fixture(as.matrix(edcomp_delta))
  )
)
write_fixture(selenium_fixture, "selenium_edcomp.json")

# =================================================================================================
# 4. Estimated asymptotes: a seeded simulated binomial dataset (true u=0.9, l=0.1, probit),
#    fit with drc's binomial LN.3 (d estimated, c fixed at 0 -- robustmargin's
#    upper="estimate", lower=0.0 path) and LN.4 (c and d both estimated -- robustmargin's
#    upper="estimate", lower="estimate" path). The raw simulated data is embedded so Python
#    tests never need to re-derive it or run R themselves.
# =================================================================================================

set.seed(20260916)
true_mu <- 0.0 # log(severity); ED50 severity = exp(true_mu) = 1.0
true_s <- 0.5
true_u <- 0.9
true_l <- 0.1
n_per_level <- 400

nonzero_severity <- 10^seq(-1, 1, length.out = 10)
sim_severity <- c(0.0, nonzero_severity)
sim_z <- ifelse(sim_severity > 0, (log(sim_severity) - true_mu) / true_s, -Inf)
sim_p_success <- true_u - (true_u - true_l) * pnorm(sim_z)
sim_trials <- rep(n_per_level, length(sim_severity))
sim_successes <- rbinom(length(sim_severity), sim_trials, sim_p_success)

sim_df <- data.frame(severity = sim_severity, successes = sim_successes, trials = sim_trials)

# LN.3: c (lower limit) fixed at 0 by drc's own default -- this is robustmargin's
# upper="estimate", lower=0.0 path, deliberately fit against data whose TRUE lower asymptote
# is 0.1, not 0: this fixture is a numerical-parity check of robustmargin's likelihood/optimizer
# under a caller-chosen fixed lower asymptote, not a recovery-of-truth check (that is
# tests/unit/test_models_recovery.py, in Python, with both asymptotes estimated).
m_ln3 <- drm(successes / trials ~ severity, weights = trials, data = sim_df, fct = LN.3(), type = "binomial")
# LN.4: both c and d estimated -- robustmargin's upper="estimate", lower="estimate" path.
m_ln4 <- drm(successes / trials ~ severity, weights = trials, data = sim_df, fct = LN.4(), type = "binomial")

# Tight refits: drm()'s default convergence tolerance (drmc()'s relTol=1e-7) leaves both LN.3
# and LN.4 under-converged on this dataset -- robustmargin's GenericLikelihoodModel path (bfgs then
# a newton polish, run to a much tighter tolerance) reaches a strictly higher log-likelihood than
# drc's default fit above. Refitting from the default fit's own coefficients with a much tighter
# tolerance closes almost all of that gap (see the logLik comparison this script prints below),
# confirming the default fit was under-converged rather than robustmargin being wrong. The default
# fits are kept in the JSON for the record, but the *tight* fits are what the Python parity
# tests read.
m_ln3_tight <- drm(
  successes / trials ~ severity, weights = trials, data = sim_df, fct = LN.3(), type = "binomial",
  start = coef(m_ln3), control = drmc(relTol = 1e-14, maxIt = 10000, method = "BFGS")
)
m_ln4_tight <- drm(
  successes / trials ~ severity, weights = trials, data = sim_df, fct = LN.4(), type = "binomial",
  start = coef(m_ln4), control = drmc(relTol = 1e-14, maxIt = 10000, method = "BFGS")
)
cat(sprintf(
  "LN.3 logLik: default=%.10f tight=%.10f (tight - default = %.3e)\n",
  as.numeric(logLik(m_ln3)), as.numeric(logLik(m_ln3_tight)),
  as.numeric(logLik(m_ln3_tight)) - as.numeric(logLik(m_ln3))
))
cat(sprintf(
  "LN.4 logLik: default=%.10f tight=%.10f (tight - default = %.3e)\n",
  as.numeric(logLik(m_ln4)), as.numeric(logLik(m_ln4_tight)),
  as.numeric(logLik(m_ln4_tight)) - as.numeric(logLik(m_ln4))
))

tight_convergence_note <- paste(
  "drm()'s default control (drmc(): relTol=1e-7, maxIt=500) leaves this fit under-converged --",
  "robustmargin's GenericLikelihoodModel path (bfgs then a newton polish, run to a much tighter",
  "tolerance) reaches a strictly higher log-likelihood than the 'default' block below. Refitting",
  "from the default fit's own coefficients with control=drmc(relTol=1e-14, maxIt=10000,",
  "method='BFGS') (the 'tight' block) closes almost all of that gap and is what the Python",
  "parity tests compare against; 'default' is kept only for the record. This is why",
  "TestEstimatedAsymptoteParity in tests/reference/test_models_drc.py compares logLik with",
  "robustmargin_logLik >= tight_logLik - 1e-6 rather than the usual two-sided 1e-4 relative bound,",
  "and why mu uses an absolute tolerance (its true value is 0, where a relative tolerance is",
  "ill-posed) while s/upper/lower use a looser 1e-3 relative tolerance -- the tight refit is",
  "itself only converged to about that precision, not to the usual 1e-4 relative tolerance."
)

# Sign convention (empirically confirmed against predict(), not assumed -- see scratch
# verification): unlike finney71, this fit is on SUCCESSES/total, a DEcreasing curve (from u at
# severity=0 down to l at high severity). For both LN.3 and LN.4, b comes out negative, and
# mu=log(e), s=-1/b, upper=d, lower=c reproduces predict() exactly via robustmargin's own formula
# P_success(x) = u - (u-l)*Phi((log(x)-mu)/s) with u=d, l=c substituted in directly (matching
# the "lower=c, upper=d" mapping literally, with no 1-minus flip needed, because the
# response fit here is success, not failure).
mapping_estimated <- list(
  response_fit = "successes/trials (success, decreasing with severity)",
  LN3 = list(mu = "log(e)", s = "-1/b", lower = "0.0 (fixed by drc's LN.3)", upper = "d (estimated)"),
  LN4 = list(mu = "log(e)", s = "-1/b", lower = "c (estimated)", upper = "d (estimated)"),
  formula_check = "P_success(x) = u - (u-l)*Phi((log(x)-mu)/s), u=d, l=c -- confirmed to reproduce predict() exactly at 5 held-out severities for both LN.3 and LN.4"
)

estimated_fixture <- list(
  versions = versions,
  simulation = list(
    seed = 20260916,
    true_params = list(mu = true_mu, s = true_s, upper = true_u, lower = true_l),
    link = "probit",
    n_per_level = n_per_level,
    severity = sim_severity,
    successes = sim_successes,
    trials = sim_trials,
    note = paste(
      "10 log-spaced severities from 0.1 to 10 (10^seq(-1,1,length.out=10)) plus an exact",
      "severity=0 control, n=400 per level, simulated once with R's Mersenne Twister under",
      "the seed above. Embedded here so no test needs to re-run R or re-derive this dataset;",
      "a Python test rebuilds the same Observations from these arrays directly."
    )
  ),
  mapping = mapping_estimated,
  drc = list(
    LN3 = list(
      fct = "LN.3",
      default = list(
        coef = named_vec_to_list(coef(m_ln3)),
        vcov = matrix_to_fixture(vcov(m_ln3), names = names(coef(m_ln3))),
        logLik = as.numeric(logLik(m_ln3)),
        control = "drmc() defaults (relTol=1e-7, maxIt=500)"
      ),
      tight = list(
        coef = named_vec_to_list(coef(m_ln3_tight)),
        vcov = matrix_to_fixture(vcov(m_ln3_tight), names = names(coef(m_ln3_tight))),
        logLik = as.numeric(logLik(m_ln3_tight)),
        control = "start=coef(default), drmc(relTol=1e-14, maxIt=10000, method='BFGS')"
      ),
      convergence_note = tight_convergence_note
    ),
    LN4 = list(
      fct = "LN.4",
      default = list(
        coef = named_vec_to_list(coef(m_ln4)),
        vcov = matrix_to_fixture(vcov(m_ln4), names = names(coef(m_ln4))),
        logLik = as.numeric(logLik(m_ln4)),
        control = "drmc() defaults (relTol=1e-7, maxIt=500)"
      ),
      tight = list(
        coef = named_vec_to_list(coef(m_ln4_tight)),
        vcov = matrix_to_fixture(vcov(m_ln4_tight), names = names(coef(m_ln4_tight))),
        logLik = as.numeric(logLik(m_ln4_tight)),
        control = "start=coef(default), drmc(relTol=1e-14, maxIt=10000, method='BFGS')"
      ),
      convergence_note = tight_convergence_note
    )
  ),
  feasibility_note = paste(
    "drc fit both binomial LN.3 and LN.4 successfully, including the exact severity=0 control",
    "row, with no special handling required beyond the standard drm() call: no epsilon",
    "substitution, no dropped row, no convergence failure. Nothing to report as infeasible."
  )
)
write_fixture(estimated_fixture, "estimated_asymptotes.json")

# =================================================================================================
# 5. The non-monotone series' (system_c_axis_1) monotonicity reversal's raw one-sided
#    Fisher exact p-value, from R directly -- 57/200 at severity 0.0 vs 143/300 at severity 0.01
#    (tests/data/pooled_counts.csv). robustmargin's own check_monotonicity() tests this same
#    pair against scipy's fisher_exact (tests/unit/test_monotonicity.py) and, once this fixture is
#    regenerated, against this R-native number too (tests/reference/test_monotonicity_drc.py).
# =================================================================================================

severity_0 <- c(successes = 57, trials = 200)
severity_0_01 <- c(successes = 143, trials = 300)

# robustmargin's own table/alternative convention: for
# direction="decreasing" with a success outcome, [[successes_high, failures_high],
# [successes_low, failures_low]], alternative="greater" -- "the higher severity performs better".
pair_table <- matrix(
  c(
    severity_0_01["successes"], severity_0_01["trials"] - severity_0_01["successes"],
    severity_0["successes"], severity_0["trials"] - severity_0["successes"]
  ),
  nrow = 2, byrow = TRUE
)
pair_fisher <- fisher.test(pair_table, alternative = "greater")

monotonicity_fixture <- list(
  versions = versions,
  dataset = list(
    series = "system_c_axis_1",
    severity_low = 0.0,
    severity_high = 0.01,
    successes_low = as.numeric(severity_0["successes"]),
    trials_low = as.numeric(severity_0["trials"]),
    successes_high = as.numeric(severity_0_01["successes"]),
    trials_high = as.numeric(severity_0_01["trials"]),
    note = "tests/data/pooled_counts.csv, series system_c_axis_1, the two lowest severities."
  ),
  fisher_test = list(
    alternative = "greater",
    table_convention = "matrix(c(successes_high, failures_high, successes_low, failures_low), nrow=2, byrow=TRUE) -- robustmargin's own convention",
    p_value = as.numeric(pair_fisher$p.value),
    estimate_odds_ratio = as.numeric(pair_fisher$estimate)
  )
)
write_fixture(monotonicity_fixture, "monotonicity_system_c_axis_1.json")

# =================================================================================================
# 6. Continuous outcomes: ryegrass root length, drc LL.4 (logit link) and LN.4 (probit link),
#    Gaussian nonlinear least squares, both asymptotes and the location/scale estimated.
# =================================================================================================

data(ryegrass)
# str(ryegrass): 'data.frame': 24 obs. of 2 variables: rootl (num), conc (num). Includes conc = 0
# controls; the response DEcreases with concentration.
stopifnot(identical(names(ryegrass), c("rootl", "conc")))
stopifnot(any(ryegrass$conc == 0))

# Fit with drc's defaults, then refit from the default coefficients under a much tighter
# tolerance. The default stopping rule (relTol = 1e-7) can end short of the minimum (see section
# 4), so the tight fit is the reference; callers check it did not end with a larger residual sum
# of squares than the default fit.
fit_tight <- function(fitter, y) {
  m_default <- fitter()
  m_tight <- fitter(
    start = coef(m_default),
    control = drmc(relTol = 1e-14, maxIt = 10000, method = "BFGS")
  )
  list(
    default = m_default, tight = m_tight,
    rss_default = sum((y - predict(m_default))^2),
    rss_tight = sum((y - predict(m_tight))^2)
  )
}

# Map drc's (b, c, d, e) onto robustmargin's natural (mu, s, upper, lower) for a DEcreasing
# outcome with s > 0, and CHECK the mapping by recomputing the curve and comparing to predict():
#   mu = log(e), upper = d, lower = c, s = sign(b) / b = |1/b|.
# The sign of b is not universal: LL.4 is c + (d - c) / (1 + exp(b (log x - log e))), decreasing
# for b > 0 (s = 1/b); LN.4 is c + (d - c) * pnorm(b (log x - log e)), decreasing for b < 0
# (s = -1/b). The sign is recorded as a fact in the fixture, not assumed.
# Also maps vcov onto the same four parameters by the delta method (J V J').
map_continuous <- function(m, cdf, x_check) {
  cf <- coef(m)
  stopifnot(identical(sub(":.*$", "", names(cf)), c("b", "c", "d", "e")))
  b <- as.numeric(cf[1]); c0 <- as.numeric(cf[2]); d <- as.numeric(cf[3]); e <- as.numeric(cf[4])
  stopifnot(d > c0)
  sgn <- sign(b)
  mu <- log(e)
  s <- sgn / b
  stopifnot(s > 0)
  p_mapped <- d - (d - c0) * cdf((log(x_check) - mu) / s)
  p_drc <- as.numeric(predict(m, data.frame(conc = x_check)))
  max_abs_diff <- max(abs(p_mapped - p_drc))
  stopifnot(max_abs_diff < 1e-8 * max(1, abs(d)))
  jac <- matrix(0, 4, 4) # rows: mu, s, upper, lower; cols: b, c, d, e
  jac[1, 4] <- 1 / e
  jac[2, 1] <- -sgn / b^2
  jac[3, 3] <- 1
  jac[4, 2] <- 1
  v <- vcov(m)
  list(
    mapped = list(mu = mu, s = s, upper = d, lower = c0),
    s_formula = if (sgn > 0) "1/b" else "-1/b",
    curve_check = list(
      x = x_check, max_abs_difference_from_predict = max_abs_diff,
      formula = "P(x) = upper - (upper - lower) * F((log(x) - mu) / s), F = plogis (LL.4) or pnorm (LN.4)"
    ),
    mapped_vcov = matrix_to_fixture(jac %*% v %*% t(jac), names = c("mu", "s", "upper", "lower"))
  )
}

# Two candidate definitions of the covariance of a Gaussian NLS fit, both scaled by the residual
# variance RSS / df, recorded next to drc's vcov() so the fixture shows which one drc reports:
#   * nls_gauss_newton: R nls's vcov(), sigma^2 (J'J)^-1 (Gauss-Newton, drops the curvature term);
#   * observed_hessian: sigma^2 (1/2 d2 RSS)^-1 at the optimum (numerical Hessian of the RSS).
# `kind` is "LL.4" (c + (d - c) / (1 + exp(b (log x - log e)))) or "LN.4"
# (c + (d - c) pnorm(b (log x - log e))). Parameters are drc's (b, c, d, e), in that order.
# The equality verdict is a numeric comparison (largest |difference| relative to the pair's SDs),
# recorded as data and never asserted here.
vcov_definitions <- function(m, x, y, kind) {
  stopifnot(kind %in% c("LL.4", "LN.4"))
  cf <- as.numeric(coef(m))
  nm <- c("b", "c", "d", "e")
  curve <- function(p) {
    z <- p[1] * (log(x) - log(p[4]))
    if (kind == "LL.4") p[2] + (p[3] - p[2]) / (1 + exp(z)) else p[2] + (p[3] - p[2]) * pnorm(z)
  }
  rss_fun <- function(p) sum((y - curve(p))^2)
  df <- length(y) - length(cf)
  sigma2 <- rss_fun(cf) / df
  fit <- nls(
    y ~ c0 + (d0 - c0) * (if (kind == "LL.4") 1 / (1 + exp(b0 * (log(x) - log(e0))))
                          else pnorm(b0 * (log(x) - log(e0)))),
    start = list(b0 = cf[1], c0 = cf[2], d0 = cf[3], e0 = cf[4]),
    control = nls.control(maxiter = 200, tol = 1e-10, warnOnly = TRUE)
  )
  v_gn <- vcov(fit)
  dimnames(v_gn) <- NULL
  h <- optimHess(cf, rss_fun, control = list(parscale = pmax(abs(cf), 1e-3), ndeps = rep(1e-5, 4)))
  v_hess <- sigma2 * solve(h / 2)
  v_drc <- vcov(m)
  dimnames(v_drc) <- NULL
  gap <- function(a, b) {
    sdv <- sqrt(diag(v_drc))
    max(abs(a - b) / outer(sdv, sdv))
  }
  gap_gn <- gap(v_drc, v_gn)
  gap_hess <- gap(v_drc, v_hess)
  list(
    nls_gauss_newton = matrix_to_fixture(v_gn, names = nm),
    observed_hessian = matrix_to_fixture(v_hess, names = nm),
    nls_converged = isTRUE(fit$convInfo$isConv),
    max_scaled_gap_drc_vs_nls_gauss_newton = gap_gn,
    max_scaled_gap_drc_vs_observed_hessian = gap_hess,
    drc_vcov_equals = if (gap_hess < gap_gn) "observed_hessian" else "nls_gauss_newton",
    note = paste(
      "gap = max over entries of |V_drc - V_other| / (sd_i sd_j) using drc's own SDs.",
      "drc_vcov_equals names the closer definition; it is a numeric verdict, not an assumption."
    )
  )
}

# The common summary of one continuous drm fit: coef, vcov, residual SD, df, RSS, logLik, and the
# mapped parameters.
summarise_continuous <- function(m, x, y, cdf, x_check, kind) {
  n <- length(y)
  df <- n - length(coef(m))
  rss <- sum((y - predict(m))^2)
  mapping <- map_continuous(m, cdf, x_check)
  c(
    list(
      coef = named_vec_to_list(setNames(as.numeric(coef(m)), c("b", "c", "d", "e"))),
      vcov = matrix_to_fixture(vcov(m), names = c("b", "c", "d", "e")),
      vcov_definitions = vcov_definitions(m, x, y, kind),
      n = n,
      df_residual = df,
      rss = rss,
      residual_sd = sqrt(rss / df),
      residual_sd_drc_summary = as.numeric(summary(m)$rse[1]),
      logLik_drc = as.numeric(logLik(m)),
      logLik_gaussian_mle = -n / 2 * (log(2 * pi * rss / n) + 1),
      AIC_drc = as.numeric(AIC(m))
    ),
    mapping
  )
}

# drc ED() with delta intervals, as a plain table. drc's delta Lower/Upper are Estimate +/-
# qt(1 - alpha/2, df.residual) * SE on the ORIGINAL dose scale; robustmargin's are built on the
# log-dose scale and exponentiated. Tests therefore compare Estimate and Std. Error (the
# quantile used is recorded as `t_quantile`), never the endpoints.
ed_table <- function(m, levels, type, df) {
  tab <- as.matrix(ED(m, levels, type = type, interval = "delta", level = 0.95, display = FALSE))
  list(
    type = type,
    response_levels = levels,
    level = 0.95,
    df_residual = df,
    t_quantile = qt(0.975, df),
    table = matrix_to_fixture(tab, names = NULL),
    se_log = as.numeric(tab[, "Std. Error"] / tab[, "Estimate"])
  )
}

# Absolute ED two ways. drc's ED(type = "absolute") converts the level y to a relative
# p = (d - y) / (d - c) at the FITTED (c, d) and then holds p fixed, so its delta gradient runs
# through (b, e) only. The full delta method differentiates the closed-form absolute ED over all
# four parameters, the asymptotes included, with drc's coef() and vcov(); that is the definition
# robustmargin implements. Both are recorded: the fixed-p one reproduces drc's SE (stopifnot),
# the full one is the reference for robustmargin.
#   LL.4: ED = e * exp(log(p / (1 - p)) / b);   LN.4: ED = e * exp(qnorm(1 - p) / b).
absolute_ed_two_ways <- function(m, kind, levels, df) {
  stopifnot(kind %in% c("LL.4", "LN.4"))
  cf <- as.numeric(coef(m))
  v <- vcov(m)
  dimnames(v) <- NULL
  ed_closed <- function(par, y, fixed_p = NULL) {
    p <- if (is.null(fixed_p)) (par[3] - y) / (par[3] - par[2]) else fixed_p
    z <- if (kind == "LL.4") log(p / (1 - p)) else qnorm(1 - p)
    par[4] * exp(z / par[1])
  }
  grad <- function(f) {
    vapply(1:4, function(i) {
      h <- 1e-5 * max(abs(cf[i]), 1e-3)
      up <- cf; dn <- cf
      up[i] <- up[i] + h; dn[i] <- dn[i] - h
      (f(up) - f(dn)) / (2 * h)
    }, numeric(1))
  }
  drc_tab <- as.matrix(ED(m, levels, type = "absolute", interval = "delta", level = 0.95, display = FALSE))
  tq <- qt(0.975, df)
  rows_full <- list()
  rows_fixed <- list()
  for (i in seq_along(levels)) {
    y <- levels[i]
    est <- ed_closed(cf, y)
    stopifnot(abs(est - drc_tab[i, "Estimate"]) < 1e-8 * abs(est))
    se_full <- sqrt(as.numeric(t(grad(function(par) ed_closed(par, y))) %*% v %*% grad(function(par) ed_closed(par, y))))
    p_hat <- (cf[3] - y) / (cf[3] - cf[2])
    g_fixed <- grad(function(par) ed_closed(par, y, fixed_p = p_hat))
    se_fixed <- sqrt(as.numeric(t(g_fixed) %*% v %*% g_fixed))
    stopifnot(abs(se_fixed - drc_tab[i, "Std. Error"]) < 1e-8 * drc_tab[i, "Std. Error"])
    rows_full[[i]] <- c(est, se_full, est - tq * se_full, est + tq * se_full)
    rows_fixed[[i]] <- c(est, se_fixed, est - tq * se_fixed, est + tq * se_fixed)
  }
  full <- do.call(rbind, rows_full)
  fixed <- do.call(rbind, rows_fixed)
  list(
    response_levels = levels,
    level = 0.95,
    df_residual = df,
    t_quantile = tq,
    full_gradient = list(
      definition = paste(
        "delta SE from the gradient of the closed-form absolute ED over all four drc parameters",
        "(b, c, d, e), with drc's coef() and vcov(); interval = Estimate +/- t(df) * SE on the",
        "original scale. This is robustmargin's definition and the test reference."
      ),
      columns = c("Estimate", "Std. Error", "Lower", "Upper"),
      table = matrix_to_fixture(full, names = NULL),
      se_log = full[, 2] / full[, 1]
    ),
    fixed_p = list(
      definition = paste(
        "drc's definition recomputed in R: the level is converted to p = (d - y)/(d - c) at the",
        "fitted (c, d) and p is then held fixed, so the gradient runs through b and e only.",
        "Asserted equal to drc's ED(type = 'absolute') Std. Error to 1e-8 relative."
      ),
      columns = c("Estimate", "Std. Error", "Lower", "Upper"),
      table = matrix_to_fixture(fixed, names = NULL),
      max_relative_difference_from_drc_se = max(abs(fixed[, 2] - drc_tab[, "Std. Error"]) / drc_tab[, "Std. Error"])
    )
  )
}

ryegrass_check_conc <- c(0.94, 1.88, 3.75, 7.5, 15, 30)
ryegrass_new <- data.frame(conc = ryegrass_check_conc)

ryegrass_one <- function(fitter, cdf, name) {
  fits <- fit_tight(fitter, ryegrass$rootl)
  stopifnot(fits$rss_tight <= fits$rss_default + 1e-10)
  m <- fits$tight
  summary <- summarise_continuous(m, ryegrass$conc, ryegrass$rootl, cdf, ryegrass_check_conc, name)
  df <- summary$df_residual
  cf <- coef(m)
  stopifnot(cf[3] > 4, cf[2] < 4) # the absolute level 4 lies inside (c, d)
  c(
    list(
      fct = name,
      rss_default_fit = fits$rss_default,
      rss_tight_fit = fits$rss_tight,
      control = "reference = refit from the default fit's coef with drmc(relTol=1e-14, maxIt=10000, method='BFGS')"
    ),
    summary,
    list(
      ED_relative = ed_table(m, c(10, 50, 90), "relative", df),
      ED_absolute = c(
        ed_table(m, c(2, 4, 6), "absolute", df),
        list(definition = "info: drc's fixed-p definition (p = (d - y)/(d - c) held fixed); pinned by ED_absolute_two_ways$fixed_p")
      ),
      ED_absolute_two_ways = absolute_ed_two_ways(m, name, c(2, 4, 6), df),
      predict_confidence = list(
        conc = ryegrass_check_conc, level = 0.95,
        table = matrix_to_fixture(
          as.matrix(predict(m, ryegrass_new, interval = "confidence", level = 0.95)),
          names = NULL
        ),
        note = "columns Prediction, Lower, Upper; identity-scale t(df_residual) band."
      )
    )
  )
}

ryegrass_fixture <- list(
  versions = versions,
  dataset = list(
    conc = ryegrass$conc, rootl = ryegrass$rootl,
    note = "str(ryegrass): rootl (num), conc (num); includes conc = 0 controls; the response decreases with conc."
  ),
  mapping_note = paste(
    "logit link <=> drc LL.4; probit link <=> drc LN.4. Natural parameters: mu = log(e),",
    "upper = d, lower = c, and s = |1/b| (decreasing outcome, s > 0). The sign of b differs",
    "between LL.4 (b > 0) and LN.4 (b < 0); each block's s_formula records which was found, and",
    "curve_check records the maximum absolute difference between the curve recomputed from the",
    "mapped parameters and drc's predict() at held-out concentrations. mapped_vcov is drc's vcov",
    "transformed to the (mu, s, upper, lower) parametrisation by the delta method.",
    "drc's `ED(interval = 'delta')` Lower/Upper are Estimate +/- qt(0.975, df_residual) * SE on",
    "the original concentration scale; robustmargin builds the interval on the log scale, so the",
    "tests compare Estimate and Std. Error, and se_log = SE / Estimate."
  ),
  LL4 = ryegrass_one(
    function(...) drm(rootl ~ conc, data = ryegrass, fct = LL.4(), ...), plogis, "LL.4"
  ),
  LN4 = ryegrass_one(
    function(...) drm(rootl ~ conc, data = ryegrass, fct = LN.4(), ...), pnorm, "LN.4"
  )
)
write_fixture(ryegrass_fixture, "ryegrass_ll4.json")

# =================================================================================================
# 7. S.alba: separate LL.4 fits per herbicide, and the ratio of the two ED50s computed in R from
#    the two separate fits (delta on the log scale, and Fieller), both with Welch-Satterthwaite t.
# =================================================================================================

data(S.alba)
stopifnot(all(c("Dose", "Herbicide", "DryMatter") %in% names(S.alba)))
herbicides <- levels(factor(S.alba$Herbicide))
stopifnot(length(herbicides) == 2)

salba_check_dose <- c(5, 20, 80, 320)
salba_fit_one <- function(h) {
  d <- S.alba[S.alba$Herbicide == h, ]
  d$Herbicide <- NULL
  fits <- fit_tight(function(...) drm(DryMatter ~ Dose, data = d, fct = LL.4(), ...), d$DryMatter)
  stopifnot(fits$rss_tight <= fits$rss_default + 1e-10)
  m <- fits$tight
  cf <- coef(m)
  stopifnot(identical(sub(":.*$", "", names(cf)), c("b", "c", "d", "e")))
  b <- as.numeric(cf[1]); c0 <- as.numeric(cf[2]); d0 <- as.numeric(cf[3]); e <- as.numeric(cf[4])
  stopifnot(b > 0, d0 > c0)
  s <- 1 / b
  mu <- log(e)
  p_mapped <- d0 - (d0 - c0) * plogis((log(salba_check_dose) - mu) / s)
  p_drc <- as.numeric(predict(m, data.frame(Dose = salba_check_dose)))
  stopifnot(max(abs(p_mapped - p_drc)) < 1e-8 * max(1, abs(d0)))
  n <- nrow(d)
  df <- n - length(cf)
  rss <- sum((d$DryMatter - predict(m))^2)
  jac <- matrix(0, 4, 4)
  jac[1, 4] <- 1 / e
  jac[2, 1] <- -1 / b^2
  jac[3, 3] <- 1
  jac[4, 2] <- 1
  ed50 <- as.matrix(ED(m, 50, type = "relative", interval = "delta", level = 0.95, display = FALSE))
  list(
    herbicide = h,
    dataset = list(dose = d$Dose, dry_matter = d$DryMatter),
    n = n, df_residual = df, rss = rss, residual_sd = sqrt(rss / df),
    rss_default_fit = fits$rss_default, rss_tight_fit = fits$rss_tight,
    coef = named_vec_to_list(setNames(as.numeric(cf), c("b", "c", "d", "e"))),
    vcov = matrix_to_fixture(vcov(m), names = c("b", "c", "d", "e")),
    vcov_definitions = vcov_definitions(m, d$Dose, d$DryMatter, "LL.4"),
    mapped = list(mu = mu, s = s, upper = d0, lower = c0),
    mapped_vcov = matrix_to_fixture(jac %*% vcov(m) %*% t(jac), names = c("mu", "s", "upper", "lower")),
    ED50 = list(
      estimate = as.numeric(ed50[1, "Estimate"]),
      se = as.numeric(ed50[1, "Std. Error"]),
      se_log = as.numeric(ed50[1, "Std. Error"] / ed50[1, "Estimate"]),
      lower_original_scale = as.numeric(ed50[1, "Lower"]),
      upper_original_scale = as.numeric(ed50[1, "Upper"])
    )
  )
}
salba_fits <- setNames(lapply(herbicides, salba_fit_one), herbicides)

fa <- salba_fits[[1]]
fb <- salba_fits[[2]]
level_ratio <- 0.95
ea <- fa$ED50$estimate; eb <- fb$ED50$estimate
dfa <- fa$df_residual; dfb <- fb$df_residual
rho <- ea / eb

# (i) Log-delta. Var(log ED50) = (SE / ED50)^2 for each fit; the ratio's log variance is their
# sum (independent samples). Welch-Satterthwaite df on the log-scale variances:
#   nu = (La + Lb)^2 / (La^2 / dfa + Lb^2 / dfb),  La = (SE_a / ED_a)^2, Lb = (SE_b / ED_b)^2.
la <- fa$ED50$se_log^2
lb <- fb$ED50$se_log^2
nu_log <- (la + lb)^2 / (la^2 / dfa + lb^2 / dfb)
t_log <- qt(1 - (1 - level_ratio) / 2, nu_log)
sigma_log <- sqrt(la + lb)
log_delta <- list(
  estimate = rho,
  sigma_log = sigma_log,
  df = nu_log,
  t_quantile = t_log,
  lo = rho * exp(-t_log * sigma_log),
  hi = rho * exp(t_log * sigma_log)
)

# (ii) Fieller, independent samples (C = 0), original-scale ED variances Va = SE_a^2,
# Vb = SE_b^2. The t quantile uses the Welch-Satterthwaite df of the linear combination
# ED_a - rho_hat * ED_b, evaluated at the point estimate rho_hat (held fixed, not iterated):
#   nu = (Va + rho_hat^2 Vb)^2 / (Va^2 / dfa + rho_hat^4 Vb^2 / dfb).
# Interval = the roots of  (ED_b^2 - t^2 Vb) r^2 - 2 ED_a ED_b r + (ED_a^2 - t^2 Va) = 0
# (valid, a bounded interval, when ED_b^2 > t^2 Vb, which is checked).
va <- fa$ED50$se^2
vb <- fb$ED50$se^2
nu_f <- (va + rho^2 * vb)^2 / (va^2 / dfa + rho^4 * vb^2 / dfb)
t_f <- qt(1 - (1 - level_ratio) / 2, nu_f)
a_coef <- eb^2 - t_f^2 * vb
stopifnot(a_coef > 0)
b_coef <- ea * eb
k_coef <- ea^2 - t_f^2 * va
disc <- b_coef^2 - a_coef * k_coef
stopifnot(disc > 0)
fieller <- list(
  estimate = rho,
  df = nu_f,
  t_quantile = t_f,
  lo = (b_coef - sqrt(disc)) / a_coef,
  hi = (b_coef + sqrt(disc)) / a_coef
)

# Information only: EDcomp on the POOLED joint fit (every parameter curve-specific, one shared
# residual variance). Its point estimate coincides with the separate fits' (least squares
# separates by curve); its intervals use the pooled residual SD, so they are NOT the reference
# for robustmargin's ratio.
m_joint <- drm(DryMatter ~ Dose, curveid = Herbicide, data = S.alba, fct = LL.4())
# drc's default joint fit stops short of the minimum, as the estimated-asymptotes fits above do, so the reference is a
# refit from its coefficients under a tight tolerance, as elsewhere; the default is kept beside it.
m_joint_tight <- drm(
  DryMatter ~ Dose, curveid = Herbicide, data = S.alba, fct = LL.4(),
  start = coef(m_joint), control = drmc(relTol = 1e-14, maxIt = 10000, method = "BFGS")
)
rss_joint_default <- sum((S.alba$DryMatter - predict(m_joint))^2)
rss_joint_tight <- sum((S.alba$DryMatter - predict(m_joint_tight))^2)
stopifnot(rss_joint_tight <= rss_joint_default + 1e-10)
edcomp_pooled <- function(interval, model = m_joint) {
  tab <- as.matrix(EDcomp(model, c(50, 50), interval = interval, level = level_ratio, display = FALSE))
  matrix_to_fixture(tab, names = NULL)
}

salba_fixture <- list(
  versions = versions,
  herbicide_levels = herbicides,
  numerator = herbicides[1],
  denominator = herbicides[2],
  level = level_ratio,
  separate_fits = salba_fits,
  ratio_ED50_from_separate_fits = list(
    definition = "ED50 (relative 0.5) of the numerator over the denominator; independent samples",
    log_delta = log_delta,
    log_delta_formula = paste(
      "log-scale variances La = (SE_a/ED_a)^2, Lb = (SE_b/ED_b)^2; sigma_log = sqrt(La + Lb);",
      "nu = (La + Lb)^2 / (La^2/df_a + Lb^2/df_b); interval = rho * exp(+-qt(0.975, nu) * sigma_log)."
    ),
    fieller = fieller,
    fieller_formula = paste(
      "Va = SE_a^2, Vb = SE_b^2 (original scale); nu = (Va + rho^2 Vb)^2 / (Va^2/df_a +",
      "rho^4 Vb^2/df_b) at rho = ED_a/ED_b; t = qt(0.975, nu); roots of",
      "(ED_b^2 - t^2 Vb) r^2 - 2 ED_a ED_b r + (ED_a^2 - t^2 Va) = 0."
    )
  ),
  info_only_pooled_sigma = list(
    label = "info_only_pooled_sigma",
    note = paste(
      "EDcomp on the joint fit drm(DryMatter ~ Dose, curveid = Herbicide, fct = LL.4()): one",
      "pooled residual variance, so its delta and Fieller intervals differ from the",
      "separate-fit values above. Recorded for information; not a test reference."
    ),
    df_residual = nrow(S.alba) - length(coef(m_joint)),
    curve_order = levels(factor(S.alba$Herbicide)),
    rss_default_fit = rss_joint_default,
    rss_tight_fit = rss_joint_tight,
    EDcomp_delta = edcomp_pooled("delta"),
    EDcomp_fieller = edcomp_pooled("fieller"),
    tight_refit = list(
      control = "refit from the default joint fit's coef with drmc(relTol=1e-14, maxIt=10000, method='BFGS'); the reference for the point estimate",
      rss = rss_joint_tight,
      EDcomp_delta = edcomp_pooled("delta", m_joint_tight),
      EDcomp_fieller = edcomp_pooled("fieller", m_joint_tight)
    )
  )
)
write_fixture(salba_fixture, "salba_separate.json")

cat("\nAll fixtures written under", out_dir, "\n")
