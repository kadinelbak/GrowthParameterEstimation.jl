# GrowthParameterEstimation.jl

Tools for fitting growth ODE models to time‑series data, with utilities for model comparison, diagnostics, workflow ranking, and joint fitting across multiple related datasets.

Current release target: `v0.3.0`.

Changelog: see `CHANGELOG.md` for release notes, including the breaking changes in `v0.3.0`.

## Features
- Built-in growth ODEs (logistic, Gompertz, exponential variants with death/delay options).
- Single-dataset fitting and model comparison utilities.
- Multi-condition workflow APIs (`build_conditions`, `rank_models`, `run_pipeline`).
- Staged fitting pipeline (`run_staged_pipeline`) with auto-select and checkpoint/manual modes.
- Population/cell-line stage templates for inherited parameter workflows.
- Strict schema validation, QC report generation, run manifest persistence, and resume-from-stage.
- Bootstrap uncertainty summaries at stage level.
- Simulation sweep engine for scenario grids.
- Joint fitting APIs for shared-parameter multi-state/multi-dataset models.
- Analysis helpers (LOO CV, k-fold CV, sensitivity, residual diagnostics, enhanced BIC analysis).

## Installation
```julia
using Pkg
Pkg.add("GrowthParameterEstimation")
```

## Minimal Working Example

This example shows the smallest end-to-end workflow: install the package, create a canonical input table, rank a couple of models, and export the results.

```julia
using Pkg
Pkg.add("GrowthParameterEstimation")
Pkg.add("DataFrames")

using GrowthParameterEstimation
using DataFrames

df = DataFrame(
    time = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    count = [1.0, 1.4, 2.1, 2.9, 3.7, 4.2],
    error = fill(0.15, 6),
    dose = fill(0.0, 6),
    cell_line = fill("A549", 6),
    density = fill(1.0, 6),
    replicate = fill(1, 6),
    unit_time = fill("h", 6),
    unit_count = fill("count", 6),
)

cfg = default_config(output_dir = "results/minimal_example")

result = run_pipeline(
    df;
    config = cfg,
    include_models = ["logistic_growth", "gompertz_growth"],
)

println(result.ranking)
println(result.exports.summary)
```

What this does:

- Normalizes and validates the input table.
- Builds fitting conditions from the rows.
- Fits the requested models and ranks them by BIC.
- Writes outputs under `results/minimal_example/`, including tables, diagnostics, and figures.

## Expected Input Schema

The package works best when your data already uses the canonical column names below. `normalize_schema` can fill in defaults for some missing fields, but the clearest path is to provide them explicitly.

| Column | Type | Example | Meaning |
| --- | --- | --- | --- |
| `time` | `Float64` | `0.0, 1.0, 2.0` | Observation times |
| `count` | `Float64` | `1.0, 1.4, 2.1` | Observed cell count or normalized burden |
| `error` | `Float64` | `0.15` | Observation uncertainty used for weighted fits |
| `dose` | `Float64` | `0.0`, `0.8` | Applied treatment dose |
| `treatment_amount` | `Float64` or `missing` | `0.8` | Alias/partner field for dose-style treatment metadata |
| `cell_line` | `String` | `"A549"` | Cell line or biological context |
| `density` | `Float64` or `missing` | `1.0` | Seeding density or starting-density metadata |
| `replicate` | `Int` | `1` | Replicate identifier |
| `unit_time` | `String` | `"h"` | Time unit label |
| `unit_count` | `String` | `"count"` | Count unit label |

Notes:

- `dose` and `treatment_amount` are kept synchronized by `normalize_schema`, so either naming convention can be used as input.
- For staged workflows with `strict_schema=true`, include metadata such as `culture_type` and `population_type` in addition to the core columns above.
- `validate_timeseries(df)` checks finite, nonnegative, and structurally consistent time-series inputs before fitting.

## Public API

The package exports many symbols, but most users should focus on the entry points below.

### Recommended entry points

| Use case | Recommended functions | Notes |
| --- | --- | --- |
| Load or standardize data | `load_timeseries`, `normalize_schema`, `validate_timeseries` | Start here when reading CSV or cleaning column names |
| Enforce richer metadata for pipelines | `validate_required_metadata`, `validate_strict_schema` | Useful before staged or production workflows |
| Fit one model to one dataset | `run_single_fit` | Smallest fitting API |
| Compare candidate models | `compare_models`, `compare_models_dict`, `rank_models` | `rank_models` is the workflow-oriented option |
| Fit across related datasets jointly | `run_joint_fit`, `compare_joint_models_dict` | For shared-parameter multi-state or multi-dataset fits |
| Run a standard pipeline | `default_config`, `build_conditions`, `run_pipeline` | Best default for end-to-end analysis |
| Run a staged pipeline | `PipelineStage`, `default_stages`, `run_staged_pipeline` | Best for inherited-parameter multi-stage workflows |
| Export workflow artifacts | `plot_topk`, `export_results`, `save_run_manifest`, `load_run_manifest` | Produces tables, diagnostics, figures, and resume state |
| Simulate a registered model | `get_model`, `list_models`, `simulate`, `run_sweep` | Use after choosing a model specification |
| Diagnostics and validation | `generate_qc_report`, `save_qc_report`, `bootstrap_stage_uncertainty`, `leave_one_out_validation`, `residual_analysis` | For QA, uncertainty, and post-fit analysis |

### Advanced or implementation-level surfaces

These are available, but they are not the best first-stop APIs for most users:

- Raw RHS model functions such as `logistic_growth!` and `gompertz_growth!` are useful when building your own `ODEProblem`, but most users should start with `run_single_fit`, `simulate`, or `run_pipeline`.
- `register_model` and `ModelSpec` are for extending the model registry, not for routine fitting of built-in models.
- `setUpProblem`, `calculate_bic`, and `pQuickStat` are lower-level fitting/statistics helpers.
- Submodule internals and underscore-prefixed helpers such as `_stage_filter` or `_build_layout` are implementation details and should not be treated as stable public API.

## Staged Workflow Example

Use `run_staged_pipeline` when parameters from one stage should feed into later stages.

```julia
using GrowthParameterEstimation
using DataFrames

df_stage = DataFrame(
    time = vcat(collect(0.0:1.0:5.0), collect(0.0:1.0:5.0)),
    count = vcat([1.0, 1.6, 2.4, 3.1, 3.7, 4.2], [1.0, 1.3, 1.7, 2.0, 2.3, 2.5]),
    error = fill(0.15, 12),
    dose = vcat(fill(0.0, 6), fill(0.8, 6)),
    cell_line = fill("A549", 12),
    density = fill(1.0, 12),
    replicate = fill(1, 12),
    unit_time = fill("h", 12),
    unit_count = fill("count", 12),
    culture_type = fill("monoculture", 12),
    population_type = fill("naive", 12),
)

stages = [
    PipelineStage(
        "untreated_monoculture",
        "Estimate baseline untreated growth",
        row -> row[:culture_type] == "monoculture" && row[:dose] == 0.0,
        [:cell_line, :population_type, :replicate],
        ["logistic_growth"],
        Symbol[],
        Dict{Symbol,Float64}(),
        Dict{Symbol,Tuple{String,Symbol}}(),
    ),
    PipelineStage(
        "treated_monoculture",
        "Fit treatment response while inheriting untreated growth parameters",
        row -> row[:culture_type] == "monoculture" && row[:dose] > 0.0,
        [:dose, :cell_line, :population_type, :replicate],
        ["theta_logistic_hill_inhibition"],
        [:ic50, :hill],
        Dict(:theta => 1.0),
        Dict(
            :r => ("untreated_monoculture", :r),
            :K => ("untreated_monoculture", :K),
        ),
    ),
]

cfg = default_config(output_dir = "results/staged_demo")

staged = run_staged_pipeline(
    df_stage;
    stages = stages,
    config = cfg,
    selection_mode = :best_bic,
    strict_schema = true,
)

println(staged.completed)
println([stage.name => stage.selected_model for stage in staged.stages])
println(keys(staged.parameter_bank))
println(staged.manifest_path)
```

Expected outputs from that staged run:

- `staged.completed == true` when every stage selects and fits a successful model.
- `staged.stages` contains one result per stage, including `status`, `candidate_models`, `selected_model`, and stage-specific output directory.
- `staged.parameter_bank` stores parameter summaries from completed stages so later stages can inherit them.
- `staged.manifest_path` points to `results/staged_demo/run_manifest.toml`, which can be used to resume or audit the run.
- When `export_stage_results=true` (the default), each stage writes ranking tables, parameter tables, failure logs, and figures under its own directory inside `results/staged_demo/`.

## Single-Fit and Comparison Examples

### Quick single-fit example
```julia
using GrowthParameterEstimation, OrdinaryDiffEq

x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [1.0, 1.8, 2.6, 3.4, 3.8]

fit = run_single_fit(x, y, [0.1, 5.0]; solver = Tsit5(), show_stats = false)
@show fit.params
@show fit.bic
@show fit.ssr
```

### Joint fit example
```julia
using GrowthParameterEstimation, OrdinaryDiffEq

function logistic_joint!(du, u, p, t)
    r, K = p
    du[1] = r * u[1] * (1 - u[1] / K)
    du[2] = r * u[2] * (1 - u[2] / K)
end

datasets = [
    (x = collect(0.0:1.0:5.0), y = [1.0, 1.4, 2.0, 2.7, 3.4, 4.0], state_index = 1),
    (x = collect(0.0:1.0:5.0), y = [2.0, 2.7, 3.8, 5.0, 6.3, 7.6], state_index = 2),
]

fit = run_joint_fit(logistic_joint!, datasets, [1.0, 2.0], [0.2, 20.0];
    solver = Tsit5(), bounds = [(0.01, 1.5), (5.0, 100.0)])

@show fit.params
@show fit.bic
```

### Compare two models
```julia
using GrowthParameterEstimation, OrdinaryDiffEq

x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [1.0, 1.8, 2.6, 3.4, 3.8]

comp = compare_models(
    x, y,
    "Logistic", logistic_growth!, [0.1, 5.0],
    "Gompertz", gompertz_growth!, [0.1, 1.0, 5.0];
    solver = Tsit5(), show_stats = false,
)

println("Best model: ", comp.best_model.name)
```

### BIC and SSR for a fixed parameter set
```julia
using GrowthParameterEstimation, DifferentialEquations, OrdinaryDiffEq

x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [1.0, 1.8, 2.6, 3.4, 3.8]

prob = ODEProblem(logistic_growth!, [y[1]], (x[1], x[end]), [0.1, 5.0])
bic, ssr = calculate_bic(prob, x, y, Tsit5(), [0.1, 5.0])
```

## Available models (in `GrowthParameterEstimation.Models`)
- `logistic_growth!(du,u,p,t)`               # p = [r, K]
- `logistic_growth_with_death!(du,u,p,t)`    # p = [r, K, death_rate]
- `gompertz_growth!(du,u,p,t)`               # p = [a, b, K]
- `gompertz_growth_with_death!(du,u,p,t)`    # p = [a, b, K, death_rate]
- `exponential_growth!(du,u,p,t)`            # p = [r]
- `exponential_growth_with_delay!(du,u,p,t)` # p = [r, K, t_lag]
- `logistic_growth_with_delay!(du,u,p,t)`    # p = [r, K, t_lag]
- `exponential_growth_with_death_and_delay!(du,u,p,t)` # p = [r, K, death_rate, t_lag]

## Key exported helpers (in `GrowthParameterEstimation`)
- Fitting: `run_single_fit`, `compare_models`, `compare_datasets`, `compare_models_dict`, `fit_three_datasets`, `run_joint_fit`, `compare_joint_models_dict`, `calculate_bic`.
- Analysis: `leave_one_out_validation`, `k_fold_cross_validation`, `parameter_sensitivity_analysis`, `residual_analysis`, `enhanced_bic_analysis`.
- Workflow/core pipeline: `run_pipeline`, `run_staged_pipeline`, `default_stages`, `default_population_stages`, `default_population_cellline_stages`, `summarize_datasets`.
- Hardening: `validate_required_metadata`, `validate_strict_schema`, `generate_qc_report`, `save_qc_report`, `save_run_manifest`, `load_run_manifest`, `bootstrap_stage_uncertainty`.
- Simulation: `simulate`, `run_sweep`.

`run_single_fit` returns a NamedTuple:
```julia
(params = Vector{Float64}, bic = Float64, ssr = Float64, solution = ODESolution)
```

## Testing
```julia
using Pkg
Pkg.test("GrowthParameterEstimation")
```

Equivalent command from repo root:

```julia
julia --project=. test/runtests.jl
```

## Interactive GUI (Web App)

If you want an interactive environment outside notebooks, use the standalone Dash app in [examples/gui/README.md](examples/gui/README.md).

It supports an end-to-end visual workflow:

- Step 1: data preflight diagnostics with actionable issues and recommendations.
- Step 2: condition construction summary.
- Step 3: model ranking plus observed-vs-predicted plotting.
- Step 4: full pipeline execution summary.
- Step 5: staged pipeline execution summary.

Quick start from repo root:

```julia
julia --project=examples/gui -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=examples/gui examples/gui/pipeline_gui_app.jl
```

Then open `http://127.0.0.1:8050`.

## Practice notebook

- One maintained practice notebook is provided at `tests/function_tour.ipynb`.
- It includes API walkthrough plus a synthetic joint-fitting example.

## Pipeline templates

- End-to-end one-shot template script: `examples/pipeline_one_shot_template.jl`.
- Step-by-step staged template notebook: `tests/pipeline_step_by_step_template.ipynb`.

## Dependencies (main)
- `DifferentialEquations.jl`
- `OrdinaryDiffEq.jl`
- `Optimization.jl`, `OptimizationOptimJL.jl`
- `DataFrames.jl`, `CSV.jl`, `StatsBase.jl`

## License
MIT (see `LICENSE`).
