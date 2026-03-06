# GrowthParameterEstimation.jl

Tools for fitting growth ODE models to time‑series data, with utilities for model comparison, diagnostics, workflow ranking, and joint fitting across multiple related datasets.

## Features
- Built-in growth ODEs (logistic, Gompertz, exponential variants with death/delay options).
- Single-dataset fitting and model comparison utilities.
- Multi-condition workflow APIs (`build_conditions`, `rank_models`, `run_pipeline`).
- Joint fitting APIs for shared-parameter multi-state/multi-dataset models.
- Analysis helpers (LOO CV, k-fold CV, sensitivity, residual diagnostics, enhanced BIC analysis).

## Installation
```julia
using Pkg
Pkg.add("GrowthParameterEstimation")
```

## Quick start
```julia
using GrowthParameterEstimation, OrdinaryDiffEq

# Data
x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [1.0, 1.8, 2.6, 3.4, 3.8]

# Fit logistic model (p = [r, K])
fit = run_single_fit(x, y, [0.1, 5.0]; solver=Tsit5(), show_stats=false)
@show fit.params  # fitted parameters
@show fit.bic     # Bayesian information criterion
@show fit.ssr     # sum of squared residuals
```

### Joint fit quick start
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
comp = compare_models(
    x, y,
    "Logistic", logistic_growth!, [0.1, 5.0],
    "Gompertz", gompertz_growth!, [0.1, 1.0, 5.0];
    solver = Tsit5(), show_stats = false
)
println("Best model: ", comp.best_model.name)
```

### BIC/SSR for a fixed parameter set
```julia
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

## Practice notebook

- One maintained practice notebook is provided at `tests/function_tour.ipynb`.
- It includes API walkthrough plus a synthetic joint-fitting example.

## Dependencies (main)
- `DifferentialEquations.jl`
- `OrdinaryDiffEq.jl`
- `Optimization.jl`, `OptimizationOptimJL.jl`
- `DataFrames.jl`, `CSV.jl`, `StatsBase.jl`

## License
MIT (see `LICENSE`).
