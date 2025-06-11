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

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kadinelbak.github.io/V1SimpleODE.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kadinelbak.github.io/V1SimpleODE.jl/dev/)
[![Build Status](https://github.com/kadinelbak/V1SimpleODE.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/kadinelbak/V1SimpleODE.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/kadinelbak/V1SimpleODE.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kadinelbak/V1SimpleODE.jl)
