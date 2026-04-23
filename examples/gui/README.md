# Pipeline GUI

This folder contains a standalone Dash web app for interactive, step-by-step model fitting workflows outside Jupyter.

## Quick Start

Run from the repository root:

```bash
julia --project=examples/gui -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=examples/gui examples/gui/pipeline_gui_app.jl
```

Open <http://127.0.0.1:8050> in your browser.

If port 8050 is busy, launch on a different port:

```bash
GPE_GUI_PORT=8051 julia --project=examples/gui examples/gui/pipeline_gui_app.jl
```

## Expected Input Data

Provide a CSV path in the app. Recommended columns:

- `time`
- `count`
- `error`
- `dose` or `treatment_amount`
- `cell_line`
- `density`
- `replicate`

For staged workflows, include metadata such as `culture_type` and `population_type`.

## Workflow Steps in the App

1. Step 1: Run Preflight
- Checks data quality and stage/filter coverage.
- Reports actionable issues and recommendations.

2. Step 2: Build Conditions
- Shows how rows are grouped into fit conditions.
- Usually very fast: typically under a second for the sample CSVs, and only a few seconds for larger files.

3. Step 3: Rank Models + Plot
- Fits selected models.
- Displays ranking and failure logs.
- Renders observed vs predicted trajectories.
- Shows in-GUI residual diagnostics for the selected model/condition.
- Shows parameter sensitivity analysis (table + bar chart) directly in the app.
- This is usually the slowest interactive step.
- Runtime grows with the number of models, grouped conditions, optimization starts, and max iterations.
- With the default settings on the sample CSVs, expect anything from a few seconds to tens of seconds.

4. Step 4: Run Full Pipeline
- Executes `run_pipeline` and summarizes ranking/failures.

5. Step 5: Run Staged Pipeline
- Executes `run_staged_pipeline` with default stages.
- Summarizes stage status and selected models.

## Troubleshooting

- If you see `No such file or directory`, verify the CSV path is absolute or relative to repository root.
- If startup is slow the first time, Julia is likely downloading/precompiling dependencies.
- If model fitting fails, run Step 1 first and use preflight issues to correct schema or condition coverage.
- The GUI now shows a loading spinner while a step is running. If Step 3 feels slow, reduce the model list or lower `n-starts` before changing anything else.
- Ensure the launch command is exactly:
	`julia --project=examples/gui examples/gui/pipeline_gui_app.jl`
	(without any trailing characters such as `~`).

Test Paths
/GrowthParameterEstimation.jl/examples/gui/data/basic_pipeline.csv
/GrowthParameterEstimation.jl/examples/gui/data/staged_monoculture.csv
/GrowthParameterEstimation.jl/examples/gui/data/cellline_stages.csv
/GrowthParameterEstimation.jl/examples/gui/data/coculture_stages.csv
/GrowthParameterEstimation.jl/examples/gui/data/sparse_preflight_warnings.csv