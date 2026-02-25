# GrowthParameterEstimation.jl – Assistant Reference

## Modules and exports
- `GrowthParameterEstimation.Models`: ODE right-hand-sides for growth models (logistic, Gompertz, exponential variants; with optional death/delay).
- `GrowthParameterEstimation.Fitting`: parameter estimation utilities.
- `GrowthParameterEstimation.Analysis`: simple validation/diagnostic helpers.
- Top-level exports:
  - Models: `logistic_growth!`, `logistic_growth_with_death!`, `gompertz_growth!`, `gompertz_growth_with_death!`, `exponential_growth!`, `exponential_growth_with_delay!`, `logistic_growth_with_delay!`, `exponential_growth_with_death_and_delay!`.
  - Fitting: `setUpProblem`, `calculate_bic`, `pQuickStat`, `run_single_fit`, `compare_models`, `compare_datasets`, `compare_models_dict`, `fit_three_datasets`.
  - Analysis: `leave_one_out_validation`, `k_fold_cross_validation`, `parameter_sensitivity_analysis`, `residual_analysis`, `enhanced_bic_analysis`.

## Core usage patterns
```julia
using GrowthParameterEstimation, OrdinaryDiffEq

# Data
x = [0.0, 1.0, 2.0, 3.0]
y = [1.0, 1.8, 2.6, 3.4]

# Fit a model
fit = run_single_fit(x, y, [0.1, 5.0]; model=logistic_growth!, solver=Tsit5())
fit.params  # vector of fitted parameters
fit.bic     # BIC
fit.ssr     # sum of squared residuals
fit.solution # ODESolution

# BIC/SSR for fixed parameters
prob = ODEProblem(logistic_growth!, [y[1]], (x[1], x[end]), [0.1, 5.0])
bic, ssr = calculate_bic(prob, x, y, Tsit5(), [0.1, 5.0])

# Compare two models
comp = compare_models(x, y,
    "Logistic", logistic_growth!, [0.1, 5.0],
    "Gompertz", gompertz_growth!, [0.1, 1.0, 5.0];
    solver = Tsit5(), show_stats = false)
comp.best_model.name
```

## Analysis helpers (all use `run_single_fit` internally)
- `leave_one_out_validation(x, y, p0; model=..., solver=..., bounds=nothing, fixed_params=nothing, show_stats=false)`
  - Returns predictions, rmse, mae, r_squared, fit_params, param_std, n_valid.
- `k_fold_cross_validation(x, y, p0; k_folds=5, model=..., solver=..., bounds=nothing, fixed_params=nothing, show_stats=false)`
  - Returns fold_metrics, overall_rmse, overall_mae, r_squared, predictions, actual.
- `parameter_sensitivity_analysis(x, y, fit_result; perturbation=0.1, model=..., solver=...)`
  - Returns sensitivity_metrics (per param), ranking, baseline_predictions, x_dense.
- `residual_analysis(x, y, fit_result; model=..., solver=..., outlier_threshold=2.0)`
  - Returns residuals, standardized_residuals, predicted_values, outlier_indices, statistics, normality_correlation, durbin_watson, autocorrelation_concern.
- `enhanced_bic_analysis(x, y; models=[...], model_names=[...], p0_values=[...], solver=Rodas5(), population_size=150, max_time=60.0)`
  - Fits each model, returns results, rankings (bic/aic/r2), bic_weights, best_model, recommendation.

## Notes and expectations
- `run_single_fit` uses `BlackBoxOptim.jl` for global search with bounds; returns only params/bic/ssr/solution.
- All solvers come from `DifferentialEquations.jl`; choose `Tsit5()` for nonstiff or `Rodas5()` for stiff problems.
- Bounds are important to prevent BlackBoxOptim from exploring pathological parameter regions.
- Tests: `Pkg.test()` runs a small smoke suite.
