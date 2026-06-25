# Architecture Context

## Stack

| Layer     | Technology                  | Role   |
| --------- | --------------------------- | ------ |
| Language  | Julia 1.6+                  | Core programming language |
| Framework | OrdinaryDiffEq, DifferentialEquations | Numerical ODE solving |
| Data      | DataFrames, CSV, StatsBase  | Data handling and statistics |
| Optimization | Optimization.jl, OptimizationOptimJL | Parameter optimization |
| Testing   | Test, Genie                 | Testing framework |
| Documentation | Manual generation        | API documentation |

## System Boundaries

- `src/` — Contains core package functionality organized by concern:
  - `data.jl` - Data loading, normalization, and validation
  - `exposure.jl` - Drug exposure modeling
  - `models.jl` - Growth model definitions
  - `registry.jl` - Model registration and management
  - `simulation.jl` - Model simulation capabilities
  - `observation.jl` - Observation modeling
  - `fitting.jl` - Parameter estimation and fitting routines
  - `analysis.jl` - Post-fit analysis and diagnostics
  - `workflow.jl` - Workflow automation and pipeline management
- `test/` — Test files and test scripts (all .jl and .ipynb test files)
- `examples/` — Example scripts
- `docs/` — Documentation files (all .md files except README.md and CHANGELOG.md)
- `context/` — Project context files (internal documentation, specs, etc.)
- `assets/` — Additional resources (images, data files, etc.)
- `results/` — Output from runs (generated, not committed)
- `log/` — Log files (generated, not committed)

## Storage Model

- **In-memory DataFrames**: Primary working data structure for time-series data
- **File System**: 
  - Results stored in `results/` directory organized by run
  - Manifests saved as TOML files for run persistence
  - Checkpoints stored for staged workflow resumption
  - Plots and diagnostic files exported as requested
- **Package Environment**: Managed through Project.toml and Manifest.toml

## Auth and Access Model

- **No authentication required**: Open-source MIT licensed package
- **Access control**: Users have full access to all exported functions
- **Usage model**: Functions accessed through `using GrowthParameterEstimation` or `GrowthParameterEstimation.function_name()`
- **Extensibility**: Users can register custom models via `register_model!()`

## Invariants

1. All exported functions maintain backward compatibility within minor versions
2. Model parameters are always validated against defined bounds when provided
3. Time-series data is validated for finite, nonnegative values before processing
4. Workflow stages maintain parameter inheritance as specified in stage definitions
5. BIC calculations follow consistent statistical formulation across all model comparisons
6. Random number generation for optimization is controllable via seed parameters where applicable
7. All public API functions return structured outputs (NamedTuples or defined types)
8. File system operations respect the configured output directory structure
