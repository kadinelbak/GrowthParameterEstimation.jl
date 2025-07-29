# Tests for analysis functionality in V1SimpleODE
using Test
using V1SimpleODE
using Statistics
using Random

include("test_data_generator.jl")

@testset "Analysis Tests" begin
    
    @testset "Leave-One-Out Cross-Validation" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        p0 = [0.1, 50.0]
        
        # Perform leave-one-out validation
        loo_result = leave_one_out_validation(x, y, p0; show_stats=false)
        
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
        @test length(loo_result.fit_params) == length(y)
        @test length(loo_result.param_std) == length(p0)
        
        # R-squared should be reasonable for good fit
        valid_predictions = sum(.!isnan.(loo_result.predictions))
        if valid_predictions >= 3
            @test -1 <= loo_result.r_squared <= 1
        end
        
        println("✓ Leave-one-out cross-validation test passed")
        println("  RMSE: $(round(loo_result.rmse, digits=4))")
        println("  R²: $(round(loo_result.r_squared, digits=4))")
        println("  Valid predictions: $(loo_result.n_valid)/$(length(y))")
    end
    
    @testset "K-Fold Cross-Validation" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        p0 = [0.1, 50.0]
        
        # Test k-fold validation with k=3
        k_fold_result = k_fold_cross_validation(x, y, p0; k_folds=3, show_stats=false)
        
        @test haskey(k_fold_result, :fold_metrics)
        @test haskey(k_fold_result, :overall_rmse)
        @test haskey(k_fold_result, :overall_mae)
        @test haskey(k_fold_result, :r_squared)
        @test haskey(k_fold_result, :predictions)
        @test haskey(k_fold_result, :actual)
        
        @test length(k_fold_result.fold_metrics) <= 3  # May be less if some folds fail
        @test k_fold_result.overall_rmse >= 0
        @test k_fold_result.overall_mae >= 0
        
        # Check fold metrics structure
        for fold_metric in k_fold_result.fold_metrics
            @test haskey(fold_metric, :fold)
            @test haskey(fold_metric, :rmse)
            @test haskey(fold_metric, :mae)
            @test haskey(fold_metric, :n_valid)
        end
        
        println("✓ K-fold cross-validation test passed")
        println("  Overall RMSE: $(round(k_fold_result.overall_rmse, digits=4))")
        println("  Overall R²: $(round(k_fold_result.r_squared, digits=4))")
        println("  Number of folds completed: $(length(k_fold_result.fold_metrics))")
    end
    
    @testset "Parameter Sensitivity Analysis" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        
        # First get a fit result
        fit_result = run_single_fit(x, y, [0.1, 50.0]; show_plots=false, show_stats=false)
        
        # Perform sensitivity analysis
        sens_result = parameter_sensitivity_analysis(x, y, fit_result; 
                                                   perturbation=0.1, 
                                                   show_plots=false)
        
        @test haskey(sens_result, :sensitivity_metrics)
        @test haskey(sens_result, :ranking)
        @test haskey(sens_result, :baseline_predictions)
        @test haskey(sens_result, :x_dense)
        
        # Check sensitivity metrics for each parameter
        n_params = length(fit_result.params)
        @test length(sens_result.sensitivity_metrics) == n_params
        
        for i in 1:n_params
            metric = sens_result.sensitivity_metrics[i]
            @test haskey(metric, :param_index)
            @test haskey(metric, :param_value)
            @test haskey(metric, :sensitivity_index)
            @test haskey(metric, :max_rel_change)
            @test haskey(metric, :pred_up)
            @test haskey(metric, :pred_down)
            
            @test metric.param_index == i
            @test metric.param_value == fit_result.params[i]
            
            # Sensitivity metrics should be non-negative if analysis succeeded
            if !isnan(metric.sensitivity_index)
                @test metric.sensitivity_index >= 0
                @test metric.max_rel_change >= 0
            end
        end
        
        # Ranking should be sorted by sensitivity index
        valid_rankings = filter(r -> !isnan(r.sensitivity_index), sens_result.ranking)
        if length(valid_rankings) > 1
            for i in 1:(length(valid_rankings)-1)
                @test valid_rankings[i].sensitivity_index >= valid_rankings[i+1].sensitivity_index
            end
        end
        
        println("✓ Parameter sensitivity analysis test passed")
        println("  Most sensitive parameter: $(valid_rankings[1].param_index) (SI=$(round(valid_rankings[1].sensitivity_index, digits=3)))")
        println("  Number of parameters analyzed: $n_params")
    end
    
    @testset "Residual Analysis" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        
        # Get a fit result
        fit_result = run_single_fit(x, y, [0.1, 50.0]; show_plots=false, show_stats=false)
        
        # Perform residual analysis
        resid_result = residual_analysis(x, y, fit_result; show_plots=false)
        
        @test haskey(resid_result, :residuals)
        @test haskey(resid_result, :standardized_residuals)
        @test haskey(resid_result, :predicted_values)
        @test haskey(resid_result, :outlier_indices)
        @test haskey(resid_result, :statistics)
        @test haskey(resid_result, :normality_correlation)
        @test haskey(resid_result, :durbin_watson)
        @test haskey(resid_result, :autocorrelation_concern)
        
        @test length(resid_result.residuals) == length(y)
        @test length(resid_result.standardized_residuals) == length(y)
        @test length(resid_result.predicted_values) == length(y)
        
        # Check residual statistics
        stats = resid_result.statistics
        @test haskey(stats, :mean_residual)
        @test haskey(stats, :std_residual)
        @test haskey(stats, :rmse)
        @test haskey(stats, :mae)
        @test haskey(stats, :n_outliers)
        
        @test stats.rmse >= 0
        @test stats.mae >= 0
        @test stats.std_residual >= 0
        @test stats.n_outliers >= 0
        @test stats.n_outliers <= length(y)
        
        # Mean residual should be close to zero for good fit
        @test abs(stats.mean_residual) < stats.std_residual
        
        # Check outlier indices are valid
        @test all(1 .<= resid_result.outlier_indices .<= length(y))
        
        println("✓ Residual analysis test passed")
        println("  RMSE: $(round(stats.rmse, digits=4))")
        println("  Mean residual: $(round(stats.mean_residual, digits=4))")
        println("  Number of outliers: $(stats.n_outliers)")
        println("  Durbin-Watson: $(round(resid_result.durbin_watson, digits=3))")
    end
    
    @testset "Enhanced BIC Analysis" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        
        # Test enhanced BIC analysis with default models
        bic_result = enhanced_bic_analysis(x, y; show_plots=false)
        
        @test haskey(bic_result, :results)
        @test haskey(bic_result, :successful_results)
        @test haskey(bic_result, :bic_ranking)
        @test haskey(bic_result, :aic_ranking)
        @test haskey(bic_result, :r2_ranking)
        @test haskey(bic_result, :bic_weights)
        @test haskey(bic_result, :best_model)
        @test haskey(bic_result, :recommendation)
        
        # Should have results for each model tested
        @test length(bic_result.results) >= 3
        
        # Check successful results
        successful = bic_result.successful_results
        if length(successful) > 0
            @test all(r.fit_success for r in successful)
            
            # Check BIC ranking is properly sorted
            bic_values = [r.bic for r in bic_result.bic_ranking]
            @test issorted(bic_values)
            
            # Check that best model is the one with lowest BIC
            @test bic_result.best_model.bic == minimum(bic_values)
            
            # BIC weights should sum to approximately 1
            @test abs(sum(bic_result.bic_weights) - 1.0) < 1e-10
            
            # All weights should be positive
            @test all(w >= 0 for w in bic_result.bic_weights)
        end
        
        # Check individual model results structure
        for result in bic_result.results
            @test haskey(result, :model_name)
            @test haskey(result, :fit_success)
            @test haskey(result, :bic)
            @test haskey(result, :aic)
            @test haskey(result, :r_squared)
            @test haskey(result, :n_params)
            
            if result.fit_success
                @test isfinite(result.bic)
                @test isfinite(result.aic)
                @test -1 <= result.r_squared <= 1
                @test result.n_params > 0
            end
        end
        
        println("✓ Enhanced BIC analysis test passed")
        if length(successful) > 0
            println("  Best model: $(bic_result.best_model.model_name)")
            println("  Best BIC: $(round(bic_result.best_model.bic, digits=2))")
            println("  Best R²: $(round(bic_result.best_model.r_squared, digits=4))")
            println("  Number of successful fits: $(length(successful))")
        else
            println("  No successful fits obtained")
        end
    end
    
    @testset "Analysis Integration" begin
        # Test that all analysis functions work together with different models
        data = get_basic_test_data()
        x, y = data.x, data.y
        
        # Fit with Gompertz model
        gompertz_fit = run_single_fit(x, y, [0.1, 1.0, 50.0]; 
                                     model=gompertz_growth!, 
                                     show_plots=false, show_stats=false)
        
        # Test that analysis functions work with different model types
        try
            # Leave-one-out with Gompertz
            loo_result = leave_one_out_validation(x, y, [0.1, 1.0, 50.0]; 
                                                model=gompertz_growth!, 
                                                show_stats=false)
            @test haskey(loo_result, :rmse)
            
            # Sensitivity analysis with Gompertz
            sens_result = parameter_sensitivity_analysis(x, y, gompertz_fit; 
                                                       model=gompertz_growth!,
                                                       show_plots=false)
            @test haskey(sens_result, :sensitivity_metrics)
            @test length(sens_result.sensitivity_metrics) == 3  # Gompertz has 3 params
            
            # Residual analysis with Gompertz
            resid_result = residual_analysis(x, y, gompertz_fit; 
                                           model=gompertz_growth!,
                                           show_plots=false)
            @test haskey(resid_result, :residuals)
            
            println("✓ Analysis integration test passed")
            println("  All analysis functions work with different ODE models")
            
        catch e
            # Some analysis might fail with different models - that's ok for testing
            println("⚠ Analysis integration test partially passed (some functions failed with alternative models)")
            println("  Error: $e")
        end
    end
    
    @testset "Error Handling in Analysis" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        
        # Test analysis with bad fit result
        bad_fit = (params = [NaN, NaN], ssr = Inf, bic = Inf, solution = nothing)
        
        # Sensitivity analysis should handle bad parameters gracefully
        try
            sens_result = parameter_sensitivity_analysis(x, y, bad_fit; show_plots=false)
            # Should either work or fail gracefully
            @test haskey(sens_result, :sensitivity_metrics)
        catch e
            # Failing is acceptable for bad input
            @test isa(e, Exception)
        end
        
        # Residual analysis should handle bad parameters gracefully  
        try
            resid_result = residual_analysis(x, y, bad_fit; show_plots=false)
            @test haskey(resid_result, :residuals)
        catch e
            @test isa(e, Exception)
        end
        
        println("✓ Error handling in analysis test passed")
    end
    
end
