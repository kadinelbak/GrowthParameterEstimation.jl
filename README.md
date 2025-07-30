# GrowthParamEst.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kadinelbak.github.io/GrowthParamEst.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kadinelbak.github.io/GrowthParamEst.jl/dev/)
[![Build Status](https://github.com/kadinelbak/GrowthParamEst.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/kadinelbak/GrowthParamEst.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/kadinelbak/GrowthParamEst.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kadinelbak/GrowthParamEst.jl)

🧬 **GrowthParamEst.jl** – A comprehensive Julia package for modeling, fitting, and analyzing ordinary differential equation (ODE) systems with applications in biological research, population dynamics, and growth modeling. V1SimpleODE.jl was the first version please understand any missed references are discussing GrowthParamEst

## 📖 Overview

GrowthParamEst.jl provides a complete toolkit for:
- **Parameter estimation** using global optimization algorithms
- **Model comparison** with statistical criteria (BIC, AIC, R²)
- **Cross-validation** and sensitivity analysis  
- **Residual diagnostics** and model validation
- **Visualization** of fits and diagnostic plots

Perfect for researchers working with biological time-series data including tumor growth, drug resistance, cell population dynamics, and any system describable by ODEs.

## 🚀 Installation

```julia
using Pkg
Pkg.add("V1SimpleODE")
```

## 📊 Data Input Format

The package expects time-series data as two separate vectors:

```julia
# Time points (independent variable)
x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]  # days, hours, etc.

# Measurements (dependent variable)  
y = [10.0, 25.0, 45.0, 70.0, 90.0, 100.0]  # cell counts, areas, concentrations, etc.
```

**Requirements:**
- `x` and `y` must be the same length
- `x` should be monotonically increasing
- `y` should contain positive values for most biological models
- Both vectors should be `Vector{<:Real}` (any numeric type)

## 🧮 Available ODE Models

V1SimpleODE includes 8 pre-built growth models:

### 1. **Logistic Growth**
```julia
logistic_growth!(du, u, p, t)
# Parameters: p = [r, K]
# r = growth rate, K = carrying capacity
```

### 2. **Logistic Growth with Death**
```julia
logistic_growth_with_death!(du, u, p, t)
# Parameters: p = [r, K, δ]
# r = growth rate, K = carrying capacity, δ = death rate
```

### 3. **Gompertz Growth**
```julia
gompertz_growth!(du, u, p, t)
# Parameters: p = [a, b, K]
# a = growth rate parameter, b = shape parameter, K = carrying capacity
```

### 4. **Gompertz Growth with Death**
```julia
gompertz_growth_with_death!(du, u, p, t)
# Parameters: p = [a, b, K, δ]
# a = growth rate, b = shape parameter, K = carrying capacity, δ = death rate
```

### 5. **Exponential Growth with Delay**
```julia
exponential_growth_with_delay!(du, u, p, t)
# Parameters: p = [r, K, t_lag]
# r = growth rate, K = carrying capacity, t_lag = delay time
```

### 6. **Logistic Growth with Delay**
```julia
logistic_growth_with_delay!(du, u, p, t)
# Parameters: p = [r, K, t_lag]
# r = growth rate, K = carrying capacity, t_lag = delay time
```

### 7. **Pure Exponential Growth**
```julia
exponential_growth!(du, u, p, t)
# Parameters: p = [r]
# r = growth rate (unbounded growth)
```

### 8. **Exponential Growth with Death and Delay**
```julia
exponential_growth_with_death_and_delay!(du, u, p, t)
# Parameters: p = [r, K, δ, t_lag]
# r = growth rate, K = carrying capacity, δ = death rate, t_lag = delay
```

## 🔧 Core Functions

### Parameter Estimation

#### `run_single_fit` - Fit Single Model
```julia
result = run_single_fit(
    x::Vector{<:Real},           # Time points
    y::Vector{<:Real},           # Measurements  
    p0::Vector{<:Real};          # Initial parameter guess
    model = logistic_growth!,     # ODE model function
    fixed_params = nothing,       # Dict to fix specific parameters
    solver = Rodas5(),           # ODE solver
    bounds = nothing,            # Parameter bounds [(min,max), ...]
    show_stats::Bool = true      # Display results
)

# Returns NamedTuple with:
# - params: fitted parameters
# - ssr: sum of squared residuals  
# - bic: Bayesian Information Criterion
# - solution: ODE solution object
```

**Example:**
```julia
using V1SimpleODE

# Your data
x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [5.0, 12.0, 28.0, 45.0, 65.0]

# Fit logistic model
result = run_single_fit(x, y, [0.1, 70.0]; 
                       model=logistic_growth!,
                       bounds=[(0.01, 1.0), (50.0, 100.0)])

println("Fitted r: $(result.params[1])")
println("Fitted K: $(result.params[2])")
println("BIC: $(result.bic)")
```

#### `fit_three_datasets` - Multiple Dataset Fitting

**Version 1: Individual datasets**
```julia
results = fit_three_datasets(
    x1, y1, "Dataset 1",
    x2, y2, "Dataset 2", 
    x3, y3, "Dataset 3",
    p0;                          # Initial parameter guess
    model = logistic_growth!,
    show_stats = true
)
```

**Version 2: Vector of datasets**
```julia
x_datasets = [x1, x2, x3]
y_datasets = [y1, y2, y3]

results = fit_three_datasets(x_datasets, y_datasets;
                            p0 = [0.1, 100.0],
                            model = logistic_growth!)

# Returns:
# - individual_fits: results for each dataset
# - summary: mean parameters, standard deviations, statistics
```

### Model Comparison

#### `compare_models` - Compare Two Models
```julia
comparison = compare_models(
    x, y,
    "Logistic", logistic_growth!, [0.1, 100.0],
    "Gompertz", gompertz_growth!, [0.1, 1.0, 100.0];
    show_stats = true,
    output_csv = "model_comparison.csv"
)

# Returns best model and comparison metrics
println("Best model: $(comparison.best_model.name)")
```

#### `compare_datasets` - Compare Same Model on Different Data  
```julia
comparison = compare_datasets(
    x1, y1, "Control",   logistic_growth!, [0.1, 100.0],
    x2, y2, "Treatment", logistic_growth!, [0.1, 100.0]
)
```

#### `compare_models_dict` - Compare Multiple Models
```julia
model_specs = Dict(
    "Logistic" => (logistic_growth!, [0.1, 100.0]),
    "Gompertz" => (gompertz_growth!, [0.1, 1.0, 100.0]),
    "Exponential" => (exponential_growth!, [0.1])
)

results = compare_models_dict(x, y, model_specs)
```

## 📈 Analysis Functions

### Cross-Validation

#### `leave_one_out_validation` - LOO Cross-Validation
```julia
loo_results = leave_one_out_validation(
    x, y, p0;
    model = logistic_growth!,
    show_stats = true
)

# Returns validation metrics:
println("RMSE: $(loo_results.rmse)")
println("R²: $(loo_results.r_squared)")  
println("Parameter std: $(loo_results.param_std)")
```

#### `k_fold_cross_validation` - K-Fold Validation
```julia
kfold_results = k_fold_cross_validation(
    x, y, p0;
    k_folds = 5,
    model = logistic_growth!
)
```

### Sensitivity and Diagnostics

#### `parameter_sensitivity_analysis` - Parameter Sensitivity
```julia
# First fit a model
fit_result = run_single_fit(x, y, [0.1, 100.0])

# Then analyze sensitivity
sens_results = parameter_sensitivity_analysis(
    x, y, fit_result;
    perturbation = 0.1,         # ±10% parameter perturbation
    show_plots = true
)

# Results include sensitivity indices and rankings
```

#### `residual_analysis` - Model Diagnostics
```julia
resid_results = residual_analysis(
    x, y, fit_result;
    outlier_threshold = 2.0,
    show_plots = true
)

# Returns comprehensive residual diagnostics:
# - residuals, standardized residuals
# - outlier detection  
# - normality tests
# - autocorrelation analysis
```

#### `enhanced_bic_analysis` - Comprehensive Model Selection
```julia
bic_results = enhanced_bic_analysis(
    x, y;
    models = [logistic_growth!, gompertz_growth!, exponential_growth!],
    model_names = ["Logistic", "Gompertz", "Exponential"],
    p0_values = [[0.1, 100.0], [0.1, 1.0, 100.0], [0.1]]
)

# Returns detailed model comparison with weights and recommendations
println("Best model: $(bic_results.best_model.model_name)")
println("Recommendation: $(bic_results.recommendation)")
```

## 🎛️ Advanced Options

### Parameter Bounds
```julia
# Constrain parameters to realistic ranges
bounds = [
    (0.01, 1.0),    # growth rate: 0.01 to 1.0
    (50.0, 200.0)   # carrying capacity: 50 to 200
]

result = run_single_fit(x, y, p0; bounds=bounds)
```

### Fixed Parameters
```julia
# Fix carrying capacity at 100, only fit growth rate
fixed_params = Dict(2 => 100.0)  # Fix parameter 2 (K) to 100.0

result = run_single_fit(x, y, [0.1, 100.0]; fixed_params=fixed_params)
```

### Custom Solvers
```julia
using DifferentialEquations

# High accuracy for stiff problems
result = run_single_fit(x, y, p0; solver=Rodas5())

# Fast solver for simple problems  
result = run_single_fit(x, y, p0; solver=Tsit5())
```

## 📋 Complete Example

```julia
using V1SimpleODE

# Generate or load your data
x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
y = [5.0, 8.0, 15.0, 28.0, 45.0, 65.0, 82.0]

# 1. Fit a single model
println("=== Single Model Fit ===")
result = run_single_fit(x, y, [0.1, 90.0]; 
                       model=logistic_growth!,
                       bounds=[(0.01, 1.0), (50.0, 150.0)])

# 2. Compare multiple models
println("\n=== Model Comparison ===")
models = Dict(
    "Logistic" => (logistic_growth!, [0.1, 90.0]),
    "Gompertz" => (gompertz_growth!, [0.1, 1.0, 90.0]),
    "Exponential+Delay" => (exponential_growth_with_delay!, [0.1, 90.0, 0.5])
)

comparison = compare_models_dict(x, y, models; show_plots=true)
println("Best model: $(comparison.best_model.name)")

# 3. Validate the best model
println("\n=== Model Validation ===")
loo_results = leave_one_out_validation(x, y, [0.1, 90.0]; 
                                      model=logistic_growth!)

# 4. Analyze parameter sensitivity  
println("\n=== Sensitivity Analysis ===")
sens_results = parameter_sensitivity_analysis(x, y, result; 
                                             show_plots=true)

# 5. Check residuals
println("\n=== Residual Analysis ===")  
resid_results = residual_analysis(x, y, result; show_plots=true)

println("\n=== Analysis Complete ===")
println("Model validation RMSE: $(round(loo_results.rmse, digits=3))")
println("Most sensitive parameter: $(sens_results.ranking[1].param_index)")
println("Number of outliers: $(resid_results.statistics.n_outliers)")
```

## 🏗️ Package Structure

```
V1SimpleODE.jl/
├── src/
│   ├── V1SimpleODE.jl     # Main module
│   ├── models.jl          # ODE model definitions  
│   ├── fitting.jl         # Parameter estimation functions
│   └── analysis.jl        # Validation and diagnostic functions
├── test/                  # Comprehensive test suite
└── docs/                  # Documentation
```

## 🧪 Testing

Run the test suite to verify installation:

```julia
using Pkg
Pkg.test("V1SimpleODE")
```

Or run individual test components:
```julia
include("test/simple_test.jl")      # Quick functionality test
include("test/test_fitting.jl")     # Fitting function tests  
include("test/test_analysis.jl")    # Analysis function tests
```

## 📚 Dependencies

Core dependencies:
- `DifferentialEquations.jl` - ODE solving
- `BlackBoxOptim.jl` - Global optimization
- `StatsBase.jl` - Statistical functions
- `Plots.jl` - Visualization
- `DataFrames.jl` - Data manipulation

## 🤝 Contributing

Contributions are welcome! Please see the issues page for current needs or submit pull requests for:
- New ODE models
- Additional analysis methods
- Performance improvements
- Documentation enhancements

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Citation

If you use V1SimpleODE.jl in your research, please cite:

```bibtex
@software{v1simpleode,
  title = {V1SimpleODE.jl: A Julia Package for ODE-Based Biological Modeling},
  author = {[Your Name]},
  url = {https://github.com/kadinelbak/V1SimpleODE.jl},
  year = {2025}
}
```
