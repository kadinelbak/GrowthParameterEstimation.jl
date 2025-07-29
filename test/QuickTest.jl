#!/usr/bin/env julia

# Quick test scri    # Test fitting with simple data
    println("Testing model fitting...")
    
catch e
    println("✗ Error in testing: ", e)
end

println("Test complete!")identify issues
println("Testing V1SimpleODE module...")

try
    using V1SimpleODE
    println("✓ Module loaded successfully")
catch e
    println("✗ Error loading module: ", e)
    exit(1)
end

try
    using CSV, DataFrames
    println("✓ CSV and DataFrames loaded")
catch e
    println("✗ Error loading CSV/DataFrames: ", e)
end

try
    using DifferentialEquations
    println("✓ DifferentialEquations loaded")
catch e
    println("✗ Error loading DifferentialEquations: ", e)
end

try
    using BlackBoxOptim
    println("✓ BlackBoxOptim loaded")
catch e
    println("✗ Error loading BlackBoxOptim: ", e)
end

# Create test data
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [10.0, 20.0, 35.0, 55.0, 80.0]

println("✓ Test data created: x=$(length(x)), y=$(length(y))")

try
    end
else
    println("✗ Test data file missing: ", test_file)
end

println("Test complete!")