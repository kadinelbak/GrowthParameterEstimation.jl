# Individual Test Functions for V1SimpleODE
# These functions can be run individually for testing specific functionality

using V1SimpleODE
using Test
using Statistics
using Random

include("test_data_generator.jl")

"""
    test_package_loading()

Test that the package loads correctly and all modules are accessible.
"""
function test_package_loading()
    println("🔧 Testing package loading...")
    
    @testset "Package Loading" begin
        # Test that all main modules are accessible
        @test isdefined(V1SimpleODE, :Models)
        @test isdefined(V1SimpleODE, :Fitting) 
        @test isdefined(V1SimpleODE, :Analysis)
        
        # Test that main functions are exported
        @test isdefined(Main, :run_single_fit)
        @test isdefined(Main, :compare_models)
        @test isdefined(Main, :logistic_growth!)
        @test isdefined(Main, :leave_one_out_validation)
        @test isdefined(Main, :parameter_sensitivity_analysis)
    end
    
    println("✓ Package loading test passed")
end

"""
    test_ode_models()

Test all ODE model functions for correctness.
"""
function test_ode_models()
    println("🔧 Testing ODE models...")
    
    @testset "ODE Models" begin
        u_test = [50.0]
        t_test = 1.0
        du_test = similar(u_test)
        
        # Test logistic growth
        p_logistic = [0.2, 100.0]
        logistic_growth!(du_test, u_test, p_logistic, t_test)
        @test length(du_test) == 1
        @test isfinite(du_test[1])
        @test du_test[1] > 0  # Should be positive growth
        
        # Test logistic with death
        p_death = [0.2, 100.0, 0.01]
        logistic_growth_with_death!(du_test, u_test, p_death, t_test)
        @test length(du_test) == 1
        @test isfinite(du_test[1])
        
        # Test Gompertz growth
        p_gompertz = [0.2, 1.5, 100.0]
        gompertz_growth!(du_test, u_test, p_gompertz, t_test)
        @test length(du_test) == 1
        @test isfinite(du_test[1])
        
        # Test exponential with delay
        p_exp_delay = [0.2, 2.0, 1.0]
        exponential_growth_with_delay!(du_test, u_test, p_exp_delay, t_test)
        @test length(du_test) == 1
        @test isfinite(du_test[1])
    end
    
    println("✓ ODE models test passed")
end

"""
    test_basic_fitting()

Test basic single fitting functionality.
"""
function test_basic_fitting()
    println("🔧 Testing basic fitting...")
    
    @testset "Basic Fitting" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        p0 = [0.1, 50.0]
        bounds = [(0.01, 2.0), (10.0, 100.0)]
        
        result = run_single_fit(x, y, p0; bounds=bounds, show_stats=false)
        
        @test haskey(result, :params)
        @test haskey(result, :ssr)
        @test haskey(result, :bic)
        @test haskey(result, :solution)
        
        @test length(result.params) == 2
        @test result.ssr >= 0
        @test isfinite(result.bic)
        
        # Parameters should be reasonable
        r_fitted, K_fitted = result.params
        @test r_fitted > 0
        @test K_fitted > 0
        @test K_fitted > maximum(y) * 0.5
    end
    
    println("✓ Basic fitting test passed")
end

"""
    test_model_comparison()

Test model comparison functionality.
"""
function test_model_comparison()
    println("🔧 Testing model comparison...")
    
    @testset "Model Comparison" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        
        # Test compare_models function
        result = compare_models(
            x, y,
            "Logistic", logistic_growth!, [0.1, 50.0],
            "Gompertz", gompertz_growth!, [0.1, 1.0, 50.0];
            bounds1=[(0.01, 2.0), (10.0, 100.0)],
            bounds2=[(0.01, 2.0), (0.1, 10.0), (10.0, 100.0)],
            show_stats=false
        )
        
        @test haskey(result, :model1)
        @test haskey(result, :model2)
        @test haskey(result, :best_model)
        @test result.best_model.name in ["Logistic", "Gompertz"]
        @test result.model1.bic >= 0
        @test result.model2.bic >= 0
    end
    
    println("✓ Model comparison test passed")
end

"""
    test_cross_validation()

Test cross-validation functionality.
"""
function test_cross_validation()
    println("🔧 Testing cross-validation...")
    
    @testset "Cross-Validation" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        p0 = [0.1, 50.0]
        bounds = [(0.01, 2.0), (10.0, 100.0)]
        
        # Test leave-one-out validation
        loo_result = leave_one_out_validation(x, y, p0; bounds=bounds, show_stats=false)
        
        @test haskey(loo_result, :predictions)
        @test haskey(loo_result, :rmse)
        @test haskey(loo_result, :r_squared)
        @test length(loo_result.predictions) == length(y)
        @test loo_result.rmse >= 0
        
        # Test k-fold validation
        kfold_result = k_fold_cross_validation(x, y, p0; k_folds=3, bounds=bounds, show_stats=false)
        
        @test haskey(kfold_result, :overall_rmse)
        @test haskey(kfold_result, :r_squared)
        @test kfold_result.overall_rmse >= 0
    end
    
    println("✓ Cross-validation test passed")
end

"""
    test_sensitivity_analysis()

Test parameter sensitivity analysis.
"""
function test_sensitivity_analysis()
    println("🔧 Testing sensitivity analysis...")
    
    @testset "Sensitivity Analysis" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        p0 = [0.1, 50.0]
        bounds = [(0.01, 2.0), (10.0, 100.0)]
        
        # First get a fit result
        fit_result = run_single_fit(x, y, p0; bounds=bounds, show_stats=false)
        
        # Test sensitivity analysis
        sens_result = parameter_sensitivity_analysis(x, y, fit_result; show_plots=false)
        
        @test haskey(sens_result, :sensitivity_metrics)
        @test haskey(sens_result, :ranking)
        @test length(sens_result.sensitivity_metrics) == length(fit_result.params)
    end
    
    println("✓ Sensitivity analysis test passed")
end

"""
    test_residual_analysis()

Test residual analysis functionality.
"""
function test_residual_analysis()
    println("🔧 Testing residual analysis...")
    
    @testset "Residual Analysis" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        p0 = [0.1, 50.0]
        bounds = [(0.01, 2.0), (10.0, 100.0)]
        
        # First get a fit result
        fit_result = run_single_fit(x, y, p0; bounds=bounds, show_stats=false)
        
        # Test residual analysis
        res_result = residual_analysis(x, y, fit_result; show_plots=false)
        
        @test haskey(res_result, :residuals)
        @test haskey(res_result, :statistics)
        @test length(res_result.residuals) == length(y)
        @test res_result.statistics.rmse >= 0
    end
    
    println("✓ Residual analysis test passed")
end

"""
    test_enhanced_bic_analysis()

Test enhanced BIC analysis with multiple models.
"""
function test_enhanced_bic_analysis()
    println("🔧 Testing enhanced BIC analysis...")
    
    @testset "Enhanced BIC Analysis" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        
        # Test enhanced BIC analysis
        bic_result = enhanced_bic_analysis(x, y; show_plots=false)
        
        @test haskey(bic_result, :results)
        @test haskey(bic_result, :best_model)
        @test haskey(bic_result, :bic_ranking)
        @test length(bic_result.results) >= 1
        @test !isempty(bic_result.bic_ranking)
    end
    
    println("✓ Enhanced BIC analysis test passed")
end

"""
    test_three_datasets()

Test fitting three datasets functionality.
"""
function test_three_datasets()
    println("🔧 Testing three datasets fitting...")
    
    @testset "Three Datasets" begin
        # Generate three different datasets
        data1 = get_basic_test_data()
        data2 = generate_logistic_test_data(r=0.3, K=80.0, seed=123)
        data3 = generate_logistic_test_data(r=0.4, K=120.0, seed=456)
        
        x_list = [data1.x, data2.x, data3.x]
        y_list = [data1.y, data2.y, data3.y]
        bounds = [(0.01, 2.0), (10.0, 150.0)]
        
        result = fit_three_datasets(x_list, y_list; bounds=bounds, show_stats=false)
        
        @test haskey(result, :dataset1)
        @test haskey(result, :dataset2)
        @test haskey(result, :dataset3)
        @test haskey(result, :summary)
        
        # Each dataset should have valid results
        @test result.dataset1.ssr >= 0
        @test result.dataset2.ssr >= 0
        @test result.dataset3.ssr >= 0
    end
    
    println("✓ Three datasets test passed")
end

# Convenience function to run all tests
"""
    run_all_individual_tests()

Run all individual test functions in sequence.
"""
function run_all_individual_tests()
    println("🚀 Running all individual tests...")
    println("="^50)
    
    test_package_loading()
    test_ode_models()
    test_basic_fitting()
    test_model_comparison()
    test_cross_validation()
    test_sensitivity_analysis()
    test_residual_analysis()
    test_enhanced_bic_analysis()
    test_three_datasets()
    
    println("="^50)
    println("✅ All individual tests completed successfully!")
end

# Export all test functions for individual use
export test_package_loading, test_ode_models, test_basic_fitting, test_model_comparison,
       test_cross_validation, test_sensitivity_analysis, test_residual_analysis,
       test_enhanced_bic_analysis, test_three_datasets, run_all_individual_tests
