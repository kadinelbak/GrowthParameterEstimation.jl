# GrowthParameterEstimation.jl

## Overview

GrowthParameterEstimation.jl is a Julia package for fitting growth ODE models to time-series data. It provides tools for model comparison, diagnostics, workflow ranking, and joint fitting across multiple related datasets. The package is designed for researchers and data scientists working with biological growth data who need to estimate parameters from experimental measurements.

## Goals

1. Provide robust tools for fitting various growth models (logistic, Gompertz, exponential) to experimental time-series data
2. Enable model comparison and selection using information criteria like BIC
3. Support complex workflows including staged fitting, joint fitting across datasets, and parameter inheritance
4. Offer diagnostics and uncertainty quantification through bootstrap methods and cross-validation
5. Provide programmatic API for accessibility

## Core User Flow

1. Load and prepare time-series data with required schema (time, count, error, dose, etc.)
2. Normalize and validate the input data schema
3. Define candidate growth models for comparison
4. Run fitting pipeline to estimate parameters and rank models by BIC
5. Export results including parameter estimates, diagnostics, and visualizations
6. Optionally run staged workflows for inherited parameter estimation
7. Perform joint fitting across related datasets when appropriate
8. Conduct post-fit analysis including sensitivity analysis and residual diagnostics

## Features

### Model Fitting and Comparison

- Built-in growth ODE models (logistic, Gompertz, exponential variants with death/delay options)
- Single-dataset fitting and model comparison utilities
- Joint fitting APIs for shared-parameter multi-state/multi-dataset models
- Custom model registration and unified fitting through ModelSpec

### Workflow Automation

- Multi-condition workflow APIs (`build_conditions`, `rank_models`, `run_pipeline`)
- Staged fitting pipeline (`run_staged_pipeline`) with auto-select and checkpoint/manual modes
- Population/cell-line stage templates for inherited parameter workflows
- Strict schema validation, QC report generation, run manifest persistence, and resume-from-stage

### Analysis and Diagnostics

- Bootstrap uncertainty summaries at stage level
- Simulation sweep engine for scenario grids
- Analysis helpers (LOO CV, k-fold CV, sensitivity, residual diagnostics, enhanced BIC analysis)
- QC report generation and validation tools

### Interfaces

- Programmatic API with multiple entry points for different use cases
- Practice notebooks and pipeline templates for learning

## Scope

### In Scope

- Core parameter estimation functionality for growth models
- Workflow tools for complex multi-stage and multi-dataset analyses
- Diagnostics and uncertainty quantification methods
- Model comparison and selection mechanisms
- Both programmatic and interactive interfaces

### Out of Scope

- Data collection or experimental design tools
- Non-growth model fitting (though the framework could be extended)
- Domain-specific biological interpretation of results
- Integration with specific laboratory information management systems
