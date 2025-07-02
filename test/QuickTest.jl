#!/usr/bin/env julia

# Quick test script to identify issues
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

# Test data loading
test_dir = dirname(@__FILE__)
test_file = joinpath(test_dir, "test_data.csv")

if isfile(test_file)
    println("✓ Test data file exists")
    try
        df = CSV.read(test_file, DataFrame)
        println("✓ Test data loaded: ", size(df))
        
        # Test data extraction
        x, y = V1SimpleODE.extractData(df)
        println("✓ Data extraction works: x=$(length(x)), y=$(length(y))")
        
    catch e
        println("✗ Error processing test data: ", e)
    end
else
    println("✗ Test data file missing: ", test_file)
end

println("Test complete!")