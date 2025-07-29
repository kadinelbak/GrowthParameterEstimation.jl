# V1SimpleODE

📦 V1SimpleODE.jl – Model Fitting & Comparison Toolkit for ODE-Based Biological Systems
V1SimpleODE.jl is a Julia package designed for rapid modeling, fitting, and comparison of ordinary differential equation (ODE) systems, with a focus on biological applications like tumor growth, drug resistance, and cell population dynamics. It supports automated parameter estimation via global optimization (BlackBoxOptim), calculates statistical metrics such as BIC and SSR, and generates informative plots to visually compare model performance.

This package streamlines workflows for researchers working with time-series biological data, enabling side-by-side evaluation of competing models and datasets (e.g., resistant vs. sensitive cells). It includes:

ODE model fitting with high-precision solvers

Statistical evaluation (BIC, SSR)

Side-by-side model comparisons

Custom visualization utilities

Full CSV export of model stats for reproducibility

Perfect for computational biology, pharmacodynamics, and systems modeling where comparing mechanistic ODE models is essential.

## Data Input Format

This package expects time-series data as two separate vectors:
- `x`: Vector of time points (e.g., days)
- `y`: Vector of corresponding measurements (e.g., cell counts, areas)

## Basic Usage

```julia
using V1SimpleODE
using DifferentialEquations

# Prepare your data as vectors
x = [1.0, 2.0, 3.0, 4.0, 5.0]  # time points
y = [10.0, 25.0, 45.0, 70.0, 90.0]  # measurements

# Define your models - these are already included in the package:
# - logistic_growth!
# - logistic_growth_with_death!
# - gompertz_growth!
# - gompertz_growth_with_death!
# - exponential_growth_with_delay!
# - logistic_growth_with_delay!

# Initial parameter guess
p0 = [0.5, 100.0]           # parameter guess [r, K] for logistic model

# Optional: define parameter bounds
bounds = [(0.0, 1.5), (75.0, 125.0)]  # search range for optimization

# Optional: choose solver
solver = Rodas5()          # high-accuracy ODE solver

# Fit a single model
result = run_single_fit(x, y, p0; model=logistic_growth!, bounds=bounds, solver=solver)
```

## Main Functions
🔹 setUpProblem(model, x, y, solver, u0, p, tspan, bounds)
Optimizes parameters for a given model and returns the fitted solution and problem.

🔹 calculate_bic(prob, x, y, solver, opt_params)
Computes BIC and SSR for a solved ODE problem.

🔹 pQuickStat(x, y, params, sol, prob, bic, ssr)
Prints parameters and plots model fit against data.

🔹 run_single_fit(x, y, p0; model, fixed_params, solver, bounds, show_stats)
Fits a single model to x,y data with parameter optimization.

🔹 compare_models(x, y, name1, model1, p0_1, name2, model2, p0_2; ...)
Fits and compares two ODE models to the same dataset. Plots results and saves CSV.

🔹 compare_datasets(x1, y1, name1, model1, p0_1, x2, y2, name2, model2, p0_2; ...)
Compare models on two different datasets. Saves comparison and plots results.

🔹 compare_models_dict(x, y, specs; ...)
Compare multiple models (specified in dictionary) on the same dataset.

🔹 fit_three_datasets(x1, y1, name1, x2, y2, name2, x3, y3, name3, p0; ...)
Fits the same ODE model to three different datasets and plots all results on one plot.

## Example Usage

```julia
# Fit same model to three datasets
fit_three_datasets(
    x1, y1, "Sample 1",
    x2, y2, "Sample 2", 
    x3, y3, "Sample 3",
    [0.5, 100.0];  # initial parameter guess
    model = logistic_growth!
)

# Compare two models on the same data
compare_models(
    x, y,
    "Logistic", logistic_growth!, [0.5, 100.0],
    "Gompertz", gompertz_growth!, [0.3, 0.1]
)

# Compare same model on two datasets  
compare_datasets(
    x1, y1, "Dataset 1", logistic_growth!, [0.5, 100.0],
    x2, y2, "Dataset 2", logistic_growth!, [0.5, 100.0]
)
```

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kadinelbak.github.io/V1SimpleODE.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kadinelbak.github.io/V1SimpleODE.jl/dev/)
[![Build Status](https://github.com/kadinelbak/V1SimpleODE.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/kadinelbak/V1SimpleODE.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/kadinelbak/V1SimpleODE.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kadinelbak/V1SimpleODE.jl)
