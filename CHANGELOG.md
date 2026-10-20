# Changelog

All notable changes to robustmargin are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html). While the version is below 1.0, a
minor release may change the public API, and every such change carries a migration note.

## [Unreleased]

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
