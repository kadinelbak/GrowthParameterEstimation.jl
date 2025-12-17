# GrowthParameterEstimation.jl

Current release: `v0.2.0`.

```@meta
CurrentModule = GrowthParameterEstimation
```

Welcome to the in-progress documentation for `GrowthParameterEstimation.jl`. The package provides ODE models and utilities for fitting, comparing, and analyzing growth dynamics. The documentation site is generated with [Documenter.jl](https://juliadocs.github.io/Documenter.jl/) directly from the package sources so it can track the public API as it evolves.

## Getting Started

```julia
using GrowthParameterEstimation

# choose a model and fit it to data
x = 0.0:1.0:5.0
y = [5.0, 8.0, 12.0, 18.0, 22.0, 28.0]
result = run_single_fit(collect(x), collect(y), [0.2, 50.0]; show_stats=false)
```

## API

```@autodocs
Modules = [GrowthParameterEstimation]
```
