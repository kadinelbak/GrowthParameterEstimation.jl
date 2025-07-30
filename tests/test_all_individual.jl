# Extended test to verify all individual test functions work
cd("..")
using GrowthParamEst
cd("test")
include("individual_tests.jl")

println("🚀 Testing all individual test functions...")
println("="^60)

# Test each function individually
test_functions = [
    ("Package Loading", test_package_loading),
    ("ODE Models", test_ode_models),
    ("Basic Fitting", test_basic_fitting),
    ("Model Comparison", test_model_comparison),
    ("Cross Validation", test_cross_validation),
    ("Sensitivity Analysis", test_sensitivity_analysis),
    ("Residual Analysis", test_residual_analysis),
    ("Enhanced BIC Analysis", test_enhanced_bic_analysis),
    ("Three Datasets", test_three_datasets)
]

successful_tests = 0
total_tests = length(test_functions)

for (name, test_func) in test_functions
    println("\n🧪 Running $name test...")
    try
        test_func()
        println("✅ $name test: PASSED")
        global successful_tests += 1
    catch e
        println("❌ $name test: FAILED")
        println("   Error: $e")
    end
end

println("\n" * "="^60)
println("📊 Test Results Summary:")
println("   Successful: $successful_tests / $total_tests")
println("   Success Rate: $(round(successful_tests/total_tests*100, digits=1))%")

if successful_tests == total_tests
    println("🎉 All individual tests are working perfectly!")
else
    println("⚠️  Some tests failed - check the errors above")
end
println("="^60)
