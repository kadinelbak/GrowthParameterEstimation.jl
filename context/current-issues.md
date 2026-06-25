# Current Issues

## Known Bugs
- No known bugs reported in issue tracker (as of latest check)
- Package appears to be actively maintained with recent commits

## Technical Debt
- Legacy ODE RHS functions (logistic_growth!, gompertz_growth!, exponential_growth!, etc.) have been removed from src/models.jl export list and will be removed completely in a future version to reduce maintenance overhead
- Registry has been updated to use composable_model_spec with builder functions instead of _ode_adapter for built-in model registration

- CI configuration only tests on windows-latest - consider adding linux and macOS for broader compatibility

## Open Questions
- What is the target release version mentioned in README (v0.3.0) - has this been released yet?
- Are there any plans to expand the library of built-in growth models?
- Is there interest in adding more sophisticated uncertainty quantification methods?
- Should the GUI examples be more prominently featured or documented?
- How should the planned buildable models feature (detailed in PLAN_FOR_BUILDABLE_MODELS.md) be prioritized and implemented?

## Recent Changes
- Based on git history, recent activity includes:
  - Debug and test files added (debug_get_param.jl, final_test.jl, simple_test.jl, etc.)
  - Updates to various test files
  - Modifications to workflow.jl and models.jl
  - Addition of PLAN_FOR_BUILDABLE_MODELS.md (detailed plan for enhancing model building capabilities)
  - These suggest active development around model building workflows

## Blockers
- No obvious blockers identified
- Package depends on standard Julia ecosystem libraries which are well-maintained