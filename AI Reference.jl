```
# GrowthParameterEstimation Package - Complete Reference Guide

## Package Overview
GrowthParameterEstimation is a Julia package for fitting and analyzing ordinary differential equation (ODE) models to biological growth data. It provides parameter estimation, model comparison, and statistical validation tools.

## Installation & Basic Setup
julia
using Pkg
Pkg.add("GrowthParameterEstimation")
using GrowthParameterEstimation
```

## Core Function Signatures

### 1. Basic Model Fitting
```julia
result = run_single_fit(
    x::Vector{<:Real},           # Time points
    y::Vector{<:Real},           # Measurements  
    p0::Vector{<:Real};          # Initial parameter guess
    model = logistic_growth!,     # ODE model function
    bounds = nothing,            # Parameter bounds [(min,max), ...]
    solver = Rodas5(),           # ODE solver
    fixed_params = nothing       # Fixed parameter indices
)

# Returns NamedTuple with fields:
# - params: fitted parameters
# - ssr: sum of squared residuals  
# - bic: Bayesian Information Criterion
# - solution: ODE solution object
# - r_squared: coefficient of determination
```

### 2. Model Comparison Functions
```julia
# Compare two models on same data
compare_models(
    x::Vector{<:Real}, y::Vector{<:Real},
    name1::String, model1::Function, p0_1::Vector,
    name2::String, model2::Function, p0_2::Vector;
    bounds1 = nothing, bounds2 = nothing,
    solver = Rodas5(),
    output_csv::String = "model_comparison.csv"
)

# Compare same model on two datasets  
compare_datasets(
    x1::Vector{<:Real}, y1::Vector{<:Real}, name1::String, 
    model1::Function, p0_1::Vector,
    x2::Vector{<:Real}, y2::Vector{<:Real}, name2::String,
    model2::Function, p0_2::Vector;
    output_csv::String = "dataset_comparison.csv"
)

# Compare multiple models using dictionary
compare_models_dict(
    x::Vector{<:Real}, y::Vector{<:Real},
    specs::Dict;  # Dict with model specs
    default_solver = Rodas5(),
    output_csv::String = "models_comparison.csv"
)

# Fit same model to three datasets
results = fit_three_datasets(
    x1, y1, name1, x2, y2, name2, x3, y3, name3,
    p0::Vector{<:Real};
    model = logistic_growth!,
    bounds = nothing,
    output_csv::String = "three_datasets_comparison.csv"
)
# Returns NamedTuple with individual_fits and summary
```

### 3. Statistical Analysis Functions
```julia
# Leave-one-out cross-validation
loo_results = leave_one_out_validation(
    x::Vector{<:Real}, y::Vector{<:Real}, p0::Vector{<:Real};
    model = logistic_growth!,
    bounds = nothing,
    solver = Rodas5()
)

# K-fold cross-validation  
cv_results = k_fold_cross_validation(
    x::Vector{<:Real}, y::Vector{<:Real}, p0::Vector{<:Real};
    model = logistic_growth!,
    k::Int = 5,
    bounds = nothing,
    solver = Rodas5()
)

# Parameter sensitivity analysis
sensitivity = parameter_sensitivity_analysis(
    x::Vector{<:Real}, y::Vector{<:Real}, 
    fit_result::NamedTuple;
    model = logistic_growth!,
    solver = Rodas5(),
    perturbation::Float64 = 0.1
)

# Residual analysis for diagnostics
residuals = residual_analysis(
    x::Vector{<:Real}, y::Vector{<:Real},
    fit_result::NamedTuple;
    model = logistic_growth!,
    solver = Rodas5(),
    outlier_threshold::Float64 = 2.0
)

# Enhanced BIC analysis for model selection
bic_results = enhanced_bic_analysis(
    x::Vector{<:Real}, y::Vector{<:Real};
    models = [logistic_growth!, gompertz_growth!],
    model_names = ["Logistic", "Gompertz"],
    p0_values = [[0.1, 100.0], [0.1, 100.0]],
    solver = Rodas5()
)
```

## Available ODE Models

### Growth Models (with parameter meanings)
```julia
# 1. Logistic Growth: du/dt = r*u*(1 - u/K)
logistic_growth!(du, u, p, t)
# Parameters: p = [r, K] where r=growth rate, K=carrying capacity

# 2. Logistic Growth with Death: du/dt = r*u*(1 - u/K) - δ*u  
logistic_growth_with_death!(du, u, p, t)
# Parameters: p = [r, K, δ] where δ=death rate

# 3. Gompertz Growth: du/dt = r*u*log(K/u)
gompertz_growth!(du, u, p, t) 
# Parameters: p = [r, K] where r=growth rate, K=carrying capacity

# 4. Gompertz Growth with Death: du/dt = r*u*log(K/u) - δ*u
gompertz_growth_with_death!(du, u, p, t)
# Parameters: p = [r, K, δ] where δ=death rate

# 5. Exponential Growth: du/dt = r*u
exponential_growth!(du, u, p, t)
# Parameters: p = [r] where r=growth rate

# 6. Exponential Growth with Delay: du/dt = r*u for t > t_lag
exponential_growth_with_delay!(du, u, p, t)
# Parameters: p = [r, t_lag] where t_lag=lag time

# 7. Logistic Growth with Delay: du/dt = r*u*(1 - u/K) for t > t_lag
logistic_growth_with_delay!(du, u, p, t)
# Parameters: p = [r, K, t_lag] where t_lag=lag time

# 8. Exponential Growth with Death and Delay
exponential_growth_with_death_and_delay!(du, u, p, t)
# Parameters: p = [r, δ, t_lag] where δ=death rate, t_lag=lag time
```

## Common Usage Patterns

### Basic Workflow Example
```julia
using GrowthParameterEstimation

# Your experimental data
time_points = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
measurements = [100.0, 150.0, 230.0, 350.0, 520.0, 720.0, 850.0]

# Fit logistic growth model
result = run_single_fit(time_points, measurements, [0.5, 1000.0];
                       bounds=[(0.1, 2.0), (500.0, 2000.0)])

println("Growth rate: ", result.params[1])
println("Carrying capacity: ", result.params[2]) 
println("Model fit quality (R²): ", result.r_squared)
```

### Model Comparison Workflow
```julia
# Compare different growth models
compare_models(time_points, measurements,
               "Logistic", logistic_growth!, [0.5, 1000.0],
               "Gompertz", gompertz_growth!, [0.3, 1000.0])

# Model selection with multiple models
bic_results = enhanced_bic_analysis(time_points, measurements;
    models = [logistic_growth!, gompertz_growth!, exponential_growth!],
    model_names = ["Logistic", "Gompertz", "Exponential"],
    p0_values = [[0.5, 1000.0], [0.3, 1000.0], [0.4]]
)
```

### Validation Workflow  
```julia
# Cross-validation for model reliability
cv_results = k_fold_cross_validation(time_points, measurements, [0.5, 1000.0];
                                   model=logistic_growth!, k=5)
println("Cross-validation RMSE: ", cv_results.overall_rmse)

# Parameter sensitivity analysis
sensitivity = parameter_sensitivity_analysis(time_points, measurements, result)
println("Most sensitive parameter: ", sensitivity.ranking[1])

# Residual analysis for model diagnostics  
residuals = residual_analysis(time_points, measurements, result)
println("Number of outliers: ", length(residuals.outliers))
```

## Return Value Structures

### run_single_fit Returns:
```julia
(
    params = [fitted_parameters...],
    ssr = sum_squared_residuals,
    bic = bayesian_information_criterion, 
    solution = ode_solution_object,
    r_squared = coefficient_of_determination
)
```

### Cross-validation Returns:
```julia
(
    fold_rmse = [rmse_per_fold...],
    fold_r2 = [r2_per_fold...], 
    overall_rmse = mean_rmse,
    overall_r2 = mean_r2,
    successful_folds = number_of_successful_folds
)
```

### Sensitivity Analysis Returns:
```julia
(
    base_prediction = baseline_model_prediction,
    perturbed_predictions = predictions_with_perturbed_params,
    sensitivity_indices = sensitivity_measure_per_parameter,
    ranking = parameters_ranked_by_sensitivity
)


## Error Handling & Troubleshooting

### Common Issues:
1. **Optimization fails**: Try different initial guesses or bounds
2. **Negative BIC**: Normal for some datasets (BIC can be negative)
3. **NaN in cross-validation**: Some folds may fail; check data quality
4. **Dimension mismatch**: Ensure x and y vectors have same length

### Recommended Parameter Bounds:
julia
# For logistic growth [r, K]
bounds = [(0.01, 5.0), (maximum(y)*0.8, maximum(y)*2.0)]

# For Gompertz growth [r, K] 
bounds = [(0.01, 2.0), (maximum(y)*0.8, maximum(y)*2.0)]

# For exponential with delay [r, t_lag]
bounds = [(0.01, 3.0), (0.0, maximum(x)*0.5)]

## Dependencies & Compatibility
- Julia ≥ 1.6
- DifferentialEquations.jl for ODE solving
- BlackBoxOptim.jl for parameter optimization  
- StatsBase.jl for statistical functions
- No plotting dependencies (removed for simplicity)

---
**Note**: This reference was created for AI assistants to understand the GrowthParameterEstimation package API and usage patterns when working in new contexts.
```