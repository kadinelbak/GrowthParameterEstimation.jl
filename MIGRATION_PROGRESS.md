# Dash -> Genie/Stipple Migration Progress

## Status

- Phase 1: Complete
- Phase 2: Complete
- Phase 3: Complete
- Phase 4: Complete
- Phase 5: In progress

Generated: 2026-05-14

---

## Completed Work

### Phase 1 - Environment Setup

- Updated examples/gui/Project.toml to use Genie stack packages.
- Removed Dash dependency from the GUI project environment.

### Phase 2 - Migration Design

- Mapped Dash callback state to Genie reactive fields.
- Preserved core helper and fitting logic to avoid behavioral regressions.

### Phase 3 - Genie Skeleton

- Created examples/gui/pipeline_gui_app_genie.jl.
- Added Genie route and Stipple app model scaffold.

### Phase 4 - Tab Migration Complete

The Genie app now includes working Tab 1/3/4/5/6 flows:

- Tab 1 (Load Data): CSV path load, dataset preview, preflight quality output, trajectory overview plot.
- Tab 3 (Build Models): template loading, equation parsing/validation, model registration, persistence to gui_custom_models.toml.
- Tab 4 (Select Models): model catalog selection and equation/parameter display.
- Tab 5 (Fit and Rank): train/validation split, ranking output, fit plot, optional bootstrap interval, full pipeline run.
- Tab 6 (Staged Analysis): staged pipeline execution and stage-by-stage summaries.

Also completed during Stage 4 finish:

- Fixed Genie file parse bug in preflight issue rendering at examples/gui/pipeline_gui_app_genie.jl.
- Added legacy entrypoint forwarding so existing launch commands still work: examples/gui/pipeline_gui_app.jl now includes the Genie app and archives old Dash code in a block comment.

---

## Current Architecture

- Primary app implementation: examples/gui/pipeline_gui_app_genie.jl
- Legacy launch alias: examples/gui/pipeline_gui_app.jl
- Existing VS Code task Restart GUI App remains valid because it calls pipeline_gui_app.jl.

---

## Remaining Phase 5 Items

- Visual polish and UX consistency pass across tabs.
- Optional re-introduction of drag-and-drop upload in Genie.
- Optional extraction of large helper blocks into a dedicated module under examples/gui/src.
- End-to-end smoke tests and documentation screenshots.

---

## Validation Notes

- Static diagnostics report no syntax errors in:
  - examples/gui/pipeline_gui_app_genie.jl
  - examples/gui/pipeline_gui_app.jl
- First app launch may spend significant time precompiling Julia packages.
