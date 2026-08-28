# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

## v0.5.1 - 2026-08-28

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
