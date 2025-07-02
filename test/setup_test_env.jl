using Pkg

# Add all required packages for testing
packages = [
    "Test",
    "CSV",
    "DataFrames",
    "DifferentialEquations",
    "StatsBase",
    "Plots",
    "SciMLSensitivity",
    "LsqFit",
    "DiffEqParamEstim",
    "Optimization",
    "ForwardDiff",
    "OptimizationOptimJL",
    "OptimizationBBO",
    "BlackBoxOptim",
    "Statistics"
]

println("Installing required packages for testing...")
for pkg in packages
    println("Adding package: $pkg")
    Pkg.add(pkg)
end
println("Package installation complete!")
