# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- Classic single-population growth laws in the model registry: `exponential_growth`,
  `theta_logistic_growth` (Richards), `bertalanffy_growth`, `general_bertalanffy_growth`,
  `power_law_growth`, `exponential_linear_growth` (smooth switch, as in Benzekry et al. 2014),
  `strong_allee_growth` and `weak_allee_growth`, each tested against its closed-form solution
  or defining property.
- `showcase/`: a separate benchmark of the package on ten public tumour-growth datasets
  (cells, spheroids, mice, patients), with a data manifest, hash-checked download script and
  shared analysis recipe. No third-party data is committed.
- `classic_gompertz_growth`: two-parameter Gompertz (`r`, `K`). The older `gompertz_growth`
  spec keeps an unused `b` parameter, so its AIC/BIC carry one extra parameter penalty;
  prefer `classic_gompertz_growth` when comparing growth laws.
- `fit(...; optimizer = :transformed)`: BFGS on an unconstrained transform of the bounded
  parameters (log scale for positive bounds spanning more than two decades). On the classic
  growth laws it reaches the same optima as the default `:fminbox` and is typically 10–300x
  faster. The default is unchanged.

### Changed

- `fit` random starts are drawn on a log scale for positive parameters whose bounds span
  more than two decades (for example `K` in `(1e-3, 1e7)`), so starts also try small values.
  Results for such models can differ from 0.5.5 for the same seed.

### Fixed

- Models whose right-hand side takes no exposure argument (the composable built-ins such as
  `logistic_growth` and `gompertz_growth`) no longer throw and catch a `MethodError` on every
  right-hand-side call in `simulate`, `fit_model` and `parameter_sensitivity_analysis`; the
  method is checked once per simulation. Fitting these models is roughly 50–100x faster.

- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1) and GitHub issue templates.
- Documentation: dedicated API Reference page covering every submodule, and tutorials in the site navigation.
- Control handoff API (`controlled_model`) that packages a fitted model for
  external optimal-control and model-predictive-control tools without solving
  control problems in this package. A `ControlModel` replaces the exposure
  function with a control input `u` and carries the point estimate plus
  bootstrap parameter sets (vectors or a `bootstrap_joint_fit` result).
- `control_rhs` / `control_rhs!` for ForwardDiff-compatible continuous dynamics
  `f(x, u, p, t)`, `mpc_functions` for the `(ẋ, x, u, d, p)` / `(y, x, d, p)`
  convention of nonlinear MPC packages, and `discrete_dynamics` for a
  zero-order-hold RK4 step map with differentiable `∂F/∂x` and `∂F/∂u`.
- `InputSchedule`, `simulate_schedule` and `schedule_ensemble` for replaying a
  piecewise-constant schedule on the point estimate and every bootstrap set.
- `control_model_summary` / `save_control_model` for plain-data (JSON) export.
- Documentation page `Control handoff` and tests in `test/control_test.jl`.

### Changed

- Documentation home page no longer pins an old release number.
- CompatHelper workflow updated to the current upstream template, runs weekly and also checks `docs/`.

### Removed

- Committed temporary files, generated outputs (`output/`, `tests/outputs/`), one-off fix scripts in `tools/` and the stray root `ci.yml`. Generated outputs are now git-ignored.
- Internal planning notes moved from `docs/` to `context/notes/`.

## v0.5.5 - 2026-10-29

### Added

- `CITATION.cff` with software citation metadata.

### Changed

- Full author name in `LICENSE` and `Project.toml`.

## v0.5.4 - 2026-08-28

### Added

- Global Latin-hypercube/PRCC sensitivity analysis across explicit parameter
  ranges, with per-series and per-time-point sensitivity summaries.
- Two-parameter profile likelihood surfaces with a joint 95% confidence region
  and bound-touching diagnostic.
- Partially pooled hierarchical joint fitting across all named experimental
  groups, with shared central parameters and log-scale group effects.

### Changed

- Updated CI and documentation metadata to target the `main` branch and supported
  Julia versions.
- Updated the documentation environment compatibility to accept the current
  `0.5` release line.
- Broadened the Julia compatibility declaration to support Julia 1.10 and newer
  1.x releases, and added Julia 1.11 to the CI matrix.
- Removed the checked-in root `Manifest.toml` so package CI and downstream users
  resolve dependencies from `Project.toml` instead of an old Julia 1.10 manifest.

### Fixed

- Corrected the hierarchical identifiability regression test so grouped datasets
  are evaluated with the full two-state model state expected by their
  `state_index` values.
- Fixed hierarchical joint-fit random-effect penalty accumulation so scalar SSE
  objectives remain scalar for multiple groups.
- Renamed the identifiability regression test's local observation-map variable
  so it no longer shadows Julia's `map` function during symbolic macro
  expansion.

## v0.5.3 - 2026-08-28

### Notes

- Superseded by `v0.5.4` after Julia General Registry registration to include
  the identifiability test macro-shadowing fix required by the package CI
  matrix.

## v0.5.2 - 2026-08-28

### Notes

- Superseded by `v0.5.3` after Julia General Registry registration to include
  the hierarchical joint-fit source fix required by the package CI matrix.

## v0.5.1 - 2026-08-28

### Notes

- Superseded by `v0.5.2` after Julia General Registry registration to fix the
  package CI test matrix while preserving the v0.5 feature set.

## v0.5.0 - 2026-08-22

### Added

- Public practical-identifiability tools for multi-start generation, prediction
  Jacobians, Fisher information, profile likelihoods, bootstrap refitting, and
  synthetic parameter-recovery benchmarks.
- Structural global and local identifiability checks backed by
  `StructuralIdentifiability.jl`, with explicit observation-map validation.
- Documentation and tests for the identifiability workflow, including a
  multi-population shared-logistic example.

### Changed

- Added `StructuralIdentifiability.jl` as a direct package dependency.
- Declared support for Julia 1.12 alongside the existing Julia 1.10 support.

### Verification

- Source parsing and repository checks completed successfully. The full Julia
  test suite could not be run on this machine because Windows Application
  Control blocks Julia's precompiled package DLLs; this is an environment
  restriction, not a reported test failure.

## v0.4.1 - 2026-08-11

### Breaking changes

- Breaking: this is a pre-1.0 minor release line, so `v0.4.x` is treated as a breaking update relative to `v0.3.x` for Julia package registration and downstream compatibility expectations.
- Breaking: joint fitting now rejects failed, non-finite, negative-prediction, and failure-sentinel fits consistently during optimization and multistart ranking. Downstream validation scripts that asserted legacy finite-fit counts may need a scientific review of the new ranked model set.
- Breaking: joint BIC summaries now count optimized joint parameters directly and exclude fixed initial-time seeding states and observations absent from `dataset_specs`, which can change model ranking summaries compared with earlier downstream helper implementations.

### Added

- Added generalized joint fitting support for fixed initial times, parameterized `u0_builder` initial states, observable callbacks, trajectory-specific residual scaling, raw/scaled SSE reporting, bounded Nelder-Mead screening, multistart fitting, and one- or two-sided bound profiling.
- Added reusable joint BIC, pooling BIC, and parameter-stability summary helpers for downstream model reconciliation workflows.

### Fixed

- Corrected joint BIC parameter counting to use the optimized parameter vector length while excluding fixed initial-time seeding states and observations not present in `dataset_specs`.
- Rejected failed, non-finite, negative-prediction, and failure-sentinel joint fits consistently during joint optimization and multistart ranking.

## v0.4.0 - 2026-08-11

### Notes

- Superseded by `v0.4.1` before Julia General Registry registration to add registry-compatible standard-library compat bounds and explicit breaking-change release notes.

## v0.3.0 - 2026-04-09

### Breaking changes

- Breaking: this is a pre-1.0 minor release, so `v0.3.0` is treated as a breaking update relative to `v0.2.x` for Julia package registration and downstream compatibility expectations.
- Breaking: staged and workflow-oriented usage now relies on the stricter schema-validation and metadata helpers introduced in this release. Existing workflow inputs may need to be normalized to the canonical columns and metadata expected by `validate_strict_schema`, `build_conditions`, and `run_staged_pipeline`.
- Breaking: workflow exports are now organized into structured output directories such as `tables/`, `params/`, `diagnostics/`, and `figures/`. Downstream scripts that assumed the older flat export layout should be updated.

### Added

- Added staged pipeline execution with checkpoint/manual modes via `run_staged_pipeline`.
- Added workflow configuration, manifests, QC reporting, and resume support.
- Added population and cell-line stage templates with parameter inheritance helpers.
- Added simulation sweep utilities and joint fitting helpers for multi-state or multi-dataset workflows.
- Added bootstrap stage uncertainty summaries and stricter data/schema validation helpers.

### Notes

- For registration release notes, reference this changelog and call out the breaking items above.
