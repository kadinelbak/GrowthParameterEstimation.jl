using V1SimpleODE
using Test
using Statistics
using Random

# Import individual test functions
include("individual_tests.jl")

# Set a global random seed for reproducible tests
Random.seed!(12345)

println("Starting V1SimpleODE test suite...")
println("Testing package version: V1SimpleODE")

@testset "V1SimpleODE.jl Test Suite" begin
    
    # Run each test category using individual test functions
    @testset "Package Loading" begin
        test_package_loading()
    end
    
    @testset "ODE Models" begin
        test_ode_models()
    end
    
    @testset "Basic Fitting" begin
        test_basic_fitting()
    end
    
    @testset "Model Comparison" begin
        test_model_comparison()
    end
    
    @testset "Cross-Validation" begin
        test_cross_validation()
    end
    
    @testset "Sensitivity Analysis" begin
        test_sensitivity_analysis()
    end
    
    @testset "Residual Analysis" begin
        test_residual_analysis()
    end
    
    @testset "Enhanced BIC Analysis" begin
        test_enhanced_bic_analysis()
    end
    
    @testset "Three Datasets" begin
        test_three_datasets()
    end
    
end

println("\n" * "="^60)
println("✅ V1SimpleODE test suite completed successfully!")
println("="^60)
