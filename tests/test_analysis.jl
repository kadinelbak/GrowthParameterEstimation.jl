# Focused Analysis Tests for GrowthParamEst
# Tests specific to the Analysis module functionality

using Test
using GrowthParamEst
using Statistics
using Random

include("test_data_generator.jl")

@testset "Analysis Module Tests" begin
    
    @testset "leave_one_out_validation Function" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        p0 = [0.1, 50.0]
        bounds = [(0.01, 2.0), (10.0, 100.0)]
        
        loo_result = leave_one_out_validation(x, y, p0; bounds=bounds, show_stats=false)
        
        @test haskey(loo_result, :predictions)
        @test haskey(loo_result, :actual)
        @test haskey(loo_result, :rmse)
        @test haskey(loo_result, :mae)
        @test haskey(loo_result, :r_squared)
        @test haskey(loo_result, :fit_params)
        @test haskey(loo_result, :param_std)
        @test haskey(loo_result, :n_valid)
        
        @test length(loo_result.predictions) == length(y)
        @test length(loo_result.actual) == length(y)
        @test loo_result.rmse >= 0
        @test loo_result.mae >= 0
        @test loo_result.n_valid <= length(y)
        
        println("✓ leave_one_out_validation tests passed")
    end
    
    @testset "k_fold_cross_validation Function" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        p0 = [0.1, 50.0]
        bounds = [(0.01, 2.0), (10.0, 100.0)]
        
        # Test with 3 folds for small dataset
        kfold_result = k_fold_cross_validation(x, y, p0; k_folds=3, bounds=bounds, show_stats=false)
        
        @test haskey(kfold_result, :fold_metrics)
        @test haskey(kfold_result, :overall_rmse)
        @test haskey(kfold_result, :overall_mae)
        @test haskey(kfold_result, :r_squared)
        
        @test kfold_result.overall_rmse >= 0
        @test kfold_result.overall_mae >= 0
        @test length(kfold_result.fold_metrics) <= 3
        
        println("✓ k_fold_cross_validation tests passed")
    end
    
    @testset "parameter_sensitivity_analysis Function" begin
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
        @test haskey(sens_result, :baseline_predictions)
        @test haskey(sens_result, :x_dense)
        
        @test length(sens_result.sensitivity_metrics) == length(fit_result.params)
        @test length(sens_result.ranking) <= length(fit_result.params)
        
        # Check that sensitivity metrics are reasonable
        for metric in values(sens_result.sensitivity_metrics)
            if !isnan(metric.sensitivity_index)
                @test metric.sensitivity_index >= 0
                @test metric.param_value > 0
            end
        end
        
        println("✓ parameter_sensitivity_analysis tests passed")
    end
    
    @testset "residual_analysis Function" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        p0 = [0.1, 50.0]
        bounds = [(0.01, 2.0), (10.0, 100.0)]
        
        # First get a fit result
        fit_result = run_single_fit(x, y, p0; bounds=bounds, show_stats=false)
        
        # Test residual analysis
        res_result = residual_analysis(x, y, fit_result; show_plots=false)
        
        @test haskey(res_result, :residuals)
        @test haskey(res_result, :standardized_residuals)
        @test haskey(res_result, :predicted_values)
        @test haskey(res_result, :outlier_indices)
        @test haskey(res_result, :statistics)
        
        @test length(res_result.residuals) == length(y)
        @test length(res_result.standardized_residuals) == length(y)
        @test length(res_result.predicted_values) == length(y)
        @test res_result.statistics.rmse >= 0
        @test res_result.statistics.mae >= 0
        
        println("✓ residual_analysis tests passed")
    end
    
    @testset "enhanced_bic_analysis Function" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        
        # Test enhanced BIC analysis with default models
        bic_result = enhanced_bic_analysis(x, y; show_plots=false)
        
        @test haskey(bic_result, :results)
        @test haskey(bic_result, :successful_results)
        @test haskey(bic_result, :bic_ranking)
        @test haskey(bic_result, :best_model)
        @test haskey(bic_result, :recommendation)
        
        @test length(bic_result.results) >= 1
        @test !isempty(bic_result.bic_ranking)
        @test !isnothing(bic_result.best_model)
        
        # Test with specific models
        models = [logistic_growth!, gompertz_growth!]
        model_names = ["Logistic", "Gompertz"]
        p0_values = [[0.1, 50.0], [0.1, 1.0, 50.0]]
        
        bic_result2 = enhanced_bic_analysis(x, y; 
                                          models=models, 
                                          model_names=model_names,
                                          p0_values=p0_values,
                                          show_plots=false)
        
        @test length(bic_result2.results) == 2
        
        println("✓ enhanced_bic_analysis tests passed")
    end
    
    @testset "Cross-Validation with Different Models" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        bounds = [(0.01, 2.0), (10.0, 100.0)]
        
        # Test LOO with Gompertz model
        loo_gompertz = leave_one_out_validation(x, y, [0.1, 1.0, 50.0]; 
                                              model=gompertz_growth!,
                                              bounds=[(0.01, 2.0), (0.1, 10.0), (10.0, 100.0)], 
                                              show_stats=false)
        
        @test haskey(loo_gompertz, :rmse)
        @test loo_gompertz.rmse >= 0
        
        println("✓ Cross-validation with different models tests passed")
    end
    
    @testset "Analysis with Edge Cases" begin
        # Test with minimal data
        x_small = [0.0, 1.0, 2.0]
        y_small = [5.0, 10.0, 15.0]
        p0 = [0.1, 20.0]
        bounds = [(0.01, 2.0), (10.0, 30.0)]
        
        # LOO should still work with minimal data
        loo_small = leave_one_out_validation(x_small, y_small, p0; bounds=bounds, show_stats=false)
        @test haskey(loo_small, :rmse)
        @test loo_small.n_valid <= length(y_small)
        
        println("✓ Edge cases tests passed")
    end
    
end
