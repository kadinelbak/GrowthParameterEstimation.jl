# Quick test of fixed individual functions
cd("..")
using GrowthParameterEstimation
cd("test")
include("individual_tests.jl")

println("🔧 Testing fixed individual test functions...")

# Test the ones that had issues
println("\n1. Testing basic fitting...")
test_basic_fitting()

println("\n2. Testing cross validation...")
test_cross_validation()

println("\n3. Testing three datasets...")
test_three_datasets()

println("\n✅ Fixed tests completed!")
