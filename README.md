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


TRYING TO CONTROL FOR ALL DATA INPUT IS HARD SO I JUST ASSUMED YOU WOULD BE GIVING DAILY INFO THAT TAKES [] for x values and [] for y values. 

Also please ensure you add a code block that will input the necessary variables. 
using V1SimpleODE
using DifferentialEquations, CSV, DataFrames

# Load your data
df = CSV.read("logistic_day_averages.csv", DataFrame)

x, y = extractData(df)  # Provided by the package


# Define your models
logistic_growth(u, p, t) = p[1] * u * (1 - u / p[2])

gompertz_growth(u, p, t) = p[1] * u * log(p[2] / u)


# Initial conditions and parameter guess
u0 = [y[1]]                # initial value

p = [0.5, 100.0]           # parameter guess

tspan = (x[1], x[end])     # time span for ODE

bounds = [(0.0, 1.5), (75.0, 125.0)]  # search range for optimization

solver = Rodas5()          # high-accuracy ODE solver


🔹 extractData(df)

Extracts cleaned x and y vectors from a DataFrame with a "Day Averages" column.


x, y = extractData(df)

🔹 setUpProblem(model, x, y, solver, u0, p, tspan, bounds)
Optimizes parameters for a given model and returns the fitted solution and problem.

params, sol, prob = setUpProblem(logistic_growth, x, y, solver, u0, p, tspan, bounds)

🔹 calculate_bic(prob, x, y, solver, opt_params)
Computes BIC and SSR for a solved ODE problem.

bic, ssr = calculate_bic(prob, x, y, solver, params)

🔹 pQuickStat(x, y, params, sol, prob, bic, ssr)
Prints parameters and plots model fit against data.

pQuickStat(x, y, params, sol, prob, bic, ssr)

🔹 compareModelsBB(name1, name2, model1, model2, x, y, solver, u0, p, tspan, bounds)
Fits and compares two ODE models to the same dataset. Plots results and saves CSV.

compareModelsBB(
    "Logistic", "Gompertz",
    logistic_growth, gompertz_growth,
    x, y, solver, u0, p, tspan, bounds
)

🔹 compareCellResponseModels(label_res, x_res, y_res, model_res, label_sen, x_sen, y_sen, model_sen, solver, u0_res, u0_sen, p, tspan, bounds)
Compare responses of resistant and sensitive cells under different models. Saves comparison and plots results.

compareCellResponseModels(
    "Resistant", x, y, logistic_growth,
    "Sensitive", x1, y1, gompertz_growth,
    solver, [y[1]], [y1[1]], p, tspan, bounds
)

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kadinelbak.github.io/V1SimpleODE.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kadinelbak.github.io/V1SimpleODE.jl/dev/)
[![Build Status](https://github.com/kadinelbak/V1SimpleODE.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/kadinelbak/V1SimpleODE.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/kadinelbak/V1SimpleODE.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kadinelbak/V1SimpleODE.jl)
