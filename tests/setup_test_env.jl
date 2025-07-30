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

# Test that GrowthParamEst can be loaded
try
    using GrowthParamEst
    println("✓ GrowthParamEst loaded successfully")
catch e
    println("✗ Failed to load GrowthParamEst: $e")
end
