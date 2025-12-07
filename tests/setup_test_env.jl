using Pkg

# Add all required packages for testing
packages = [
    "Test",
    "DifferentialEquations",
    "StatsBase",
    "Plots",
    "BlackBoxOptim",
    "Statistics",
    "Random",
    "Distributions"
]

println("Installing required packages for testing...")
for pkg in packages
    try
        println("Adding package: $pkg")
        Pkg.add(pkg)
    catch e
        println("Warning: Could not add $pkg - $e")
    end
end
println("Package installation complete!")

# Test that GrowthParameterEstimation can be loaded
try
    using GrowthParameterEstimation
    println("✓ GrowthParameterEstimation loaded successfully")
catch e
    println("✗ Failed to load GrowthParameterEstimation: $e")
end
