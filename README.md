# GrowthParameterEstimation.jl

Tools for fitting simple growth ODE models to time‑series data. Provides a handful of built‑in models, a global optimizer–based fitting routine, and convenience helpers for basic model comparison and diagnostics.

## Features
- Eight built‑in growth ODEs (logistic, Gompertz, exponential variants, with optional death and delay).
- Global parameter search via `BlackBoxOptim.jl` on top of `DifferentialEquations.jl`.
- Convenience utilities for BIC/SSR calculation, model comparison, and simple cross‑validation/sensitivity checks.

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
- Fitting: `run_single_fit`, `compare_models`, `compare_datasets`, `compare_models_dict`, `fit_three_datasets`, `calculate_bic`.
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

## Dependencies (main)
- `DifferentialEquations.jl`
- `OrdinaryDiffEq.jl`
- `BlackBoxOptim.jl`
- `DataFrames.jl`, `CSV.jl`, `StatsBase.jl`

## License
MIT (see `LICENSE`).
