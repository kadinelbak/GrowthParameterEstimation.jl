# Test script to verify individual_tests.jl works properly
# This script tests the include functionality and runs a sample individual test

# Change to the parent directory to ensure proper package loading
cd("..")

# Load the package
using V1SimpleODE

# Go back to test directory
cd("test")

# Include the individual test functions
include("individual_tests.jl")

println("🔍 Testing if include worked properly...")

# Test 1: Check if functions from test_data_generator are available
try
    data = get_basic_test_data()
    println("✓ get_basic_test_data() is available and working")
    println("  Data structure: x has $(length(data.x)) points, y has $(length(data.y)) points")
catch e
    println("❌ Error accessing get_basic_test_data(): $e")
end

# Test 2: Check if individual test functions are defined
test_functions = [
    :test_package_loading,
    :test_ode_models, 
    :test_basic_fitting,
    :test_model_comparison,
    :run_all_individual_tests
]

println("\n🔍 Checking if individual test functions are defined...")
for func in test_functions
    if isdefined(Main, func)
        println("✓ $func is defined")
    else
        println("❌ $func is NOT defined")
    end
end

# Test 3: Try running the package loading test
println("\n🧪 Testing package loading function...")
try
    test_package_loading()
    println("✅ Package loading test completed successfully!")
catch e
    println("❌ Package loading test failed: $e")
end

# Test 4: Try running the ODE models test
println("\n🧪 Testing ODE models function...")
try
    test_ode_models()
    println("✅ ODE models test completed successfully!")
catch e
    println("❌ ODE models test failed: $e")
end

println("\n" * "="^50)
println("✅ Individual test verification completed!")
println("="^50)
