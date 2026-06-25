# Basic Examples

This tutorial covers common usage patterns for GrowthParameterEstimation.jl.

## Working with Different Models

The package provides several built-in growth models through the Models module:

```julia
using GrowthParameterEstimation

# Logistic growth
logistic_model = Models.build_logistic()

# Gompertz growth  
gompertz_model = Models.build_gompertz()

# Exponential growth
exponential_model = Models.build_exponential()

# Exponential growth with delay
exp_delay_model = Models.build_exponential_with_delay()
```

## Fitting with Drug Exposure

You can fit models that include drug exposure effects:

```julia
# Time points and synthetic data
t = 0.0:0.5:15.0
y = [50.0 / (1.0 + 49.0 * exp(-0.4 * ti)) for ti in t] + 0.3*randn(length(t))

# Fit with constant exposure
result = fit_model(
    Registry.get_model("logistic_growth"),
    t, y,
    dose=5.0  # 5 units of drug exposure
)

println("Growth rate with drug exposure: $(result.params[1])")
println("Carrying capacity with drug exposure: $(result.params[2])")
```

## Parameter Sweeps

Run parameter sweeps to explore model behavior:

```julia
using GrowthParameterEstimation.Simulation

# Define a parameter sweep
grid = SweepGrid(
    [1.0, 5.0, 10.0],        # seed totals
    [0.0, 0.1, 0.5],         # resistant fractions  
    [0.0, 2.5, 10.0],        # drug doses
    0.0:0.5:20.0             # time points
)

# Get model specification
spec = Registry.get_model("logistic_growth")

# Run the sweep
sweep_result = run_sweep(spec, [0.4, 80.0], grid)  # [growth rate, carrying capacity]

# Access results
summary_df = sweep_result.summary
first_result = sweep_result.simulations[(seed_total=1.0, resistant_fraction=0.0, dose=0.0)]

println("Summary of sweep results:")
show(first(summary_df, 5))  # Show first 5 rows
```

## Model Selection

Perform automated model selection using information criteria:

```julia
# Generate data from a Gompertz process
t = 0.0:0.5:12.0
y_true = [80.0 * exp(-exp(1.0 - 0.3 * ti)) for ti in t]
y = y_true + 0.8*randn(length(t))  # Add noise

# Compare multiple models
results = enhanced_bic_analysis(
    t, y;
    models = [
        Models.build_exponential(),
        Models.build_logistic(), 
        Models.build_gompertz()
    ],
    model_names = ["Exponential", "Logistic", "Gompertz"]
)

# See model rankings
println("Model rankings by BIC:")
for (i, model) in enumerate(results.bic_ranking)
    println("$i. $(model.model_name): BIC = $(round(model.bic, digits=2))")
end
```

## Cross-Validation

Evaluate model performance using cross-validation:

```julia
# Perform leave-one-out cross-validation
loo_results = leave_one_out_validation(
    t, y, [0.2, 40.0];
    model=Models.build_logistic()
)

println("LOO RMSE: $(loo_results.rmse)")
println("LOO R²: $(loo_results.r_squared)")

# Perform k-fold cross-validation  
cv_results = k_fold_cross_validation(
    t, y, [0.2, 40.0];
    model=Models.build_logistic(),
    k_folds=5
)

println("5-fold CV RMSE: $(cv_results.overall_rmse)")
println("5-fold CV R²: $(cv_results.r_squared)")
```