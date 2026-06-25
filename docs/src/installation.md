# Installation

GrowthParameterEstimation.jl is available for installation through the Julia package manager.

## Installing from the Julia Registry

To install the package, open a Julia session and run:

```julia
using Pkg
Pkg.add("GrowthParameterEstimation")
```

## Installing from Source

To install the latest development version from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/your-username/GrowthParameterEstimation.jl")
```

## Verifying Installation

After installation, verify that the package loads correctly:

```julia
using GrowthParameterEstimation
```

You should see no error messages. You can also check the version:

```julia
using Pkg
Pkg.status("GrowthParameterEstimation")
```

## Dependencies

GrowthParameterEstimation.jl has the following dependencies:
- DifferentialEquations.jl
- OrdinaryDiffEq.jl
- Optimization.jl
- DataFrames.jl
- CSV.jl
- StatsBase.jl
- LsqFit.jl
- RecursiveArrayTools.jl
- DiffEqParamEstim.jl
- ForwardDiff.jl
- OptimizationOptimJL.jl
- TOML.jl
- Distributions.jl
- Random.jl

These will be automatically installed when you add the package.