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

# Test that V1SimpleODE can be loaded
try
    using V1SimpleODE
    println("✓ V1SimpleODE loaded successfully")
catch e
    println("✗ Failed to load V1SimpleODE: $e")
end
