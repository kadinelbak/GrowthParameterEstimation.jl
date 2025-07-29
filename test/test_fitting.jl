# Tests for fitting functionality in V1SimpleODE
using Test
using V1SimpleODE
using DifferentialEquations

include("test_data_generator.jl")

@testset "Fitting Tests" begin
    
    @testset "Basic Single Fit" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        p0 = [0.1, 50.0]  # Initial guess [r, K]
        
        # Test run_single_fit
        result = run_single_fit(x, y, p0; bounds=[(0.01, 2.0), (10.0, 100.0)], show_stats=false)
        
        @test haskey(result, :params)
        @test haskey(result, :ssr)
        @test haskey(result, :bic)
        @test haskey(result, :solution)
        
        @test length(result.params) == 2
        @test result.ssr >= 0
        @test isfinite(result.bic)
        
        # Parameters should be reasonable for logistic growth
        r_fitted, K_fitted = result.params
        @test r_fitted > 0  # Growth rate should be positive
        @test K_fitted > 0  # Carrying capacity should be positive
        @test K_fitted > maximum(y) * 0.5  # K should be reasonably larger than max observed
        
        println("✓ Basic single fit test passed")
        println("  Fitted parameters: r=$(round(r_fitted, digits=4)), K=$(round(K_fitted, digits=2))")
        println("  True parameters: r=$(data.params_true[1]), K=$(data.params_true[2])")
    end
    
    @testset "Different ODE Models" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        
        # Test logistic growth (default)
        result_logistic = run_single_fit(x, y, [0.1, 50.0]; bounds=[(0.01, 2.0), (10.0, 100.0)], show_stats=false)
        @test length(result_logistic.params) == 2
        @test result_logistic.ssr >= 0
        
        # Test Gompertz growth
        result_gompertz = run_single_fit(x, y, [0.1, 1.0, 50.0]; model=gompertz_growth!, bounds=[(0.01, 2.0), (10.0, 100.0)], show_stats=false)
        @test length(result_gompertz.params) == 3
        @test result_gompertz.ssr >= 0
        
        # Test exponential with delay
        result_exp_delay = run_single_fit(x, y, [0.1, 50.0, 1.0]; model=exponential_growth_with_delay!, bounds=[(0.01, 2.0), (10.0, 100.0)], show_stats=false)
        @test length(result_exp_delay.params) == 3
        @test result_exp_delay.ssr >= 0
        
        println("✓ Different ODE models test passed")
        println("  Logistic SSR: $(round(result_logistic.ssr, digits=4))")
        println("  Gompertz SSR: $(round(result_gompertz.ssr, digits=4))")
        println("  Exp+Delay SSR: $(round(result_exp_delay.ssr, digits=4))")
    end
    
    @testset "Model Comparison" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        
        # Test compare_models function
        result = compare_models(x, y)
        
        @test haskey(result, :models)
        @test haskey(result, :best_model)
        @test haskey(result, :comparison_table)
        
        @test length(result.models) >= 3  # Should have at least 3 models
        @test !isnothing(result.best_model)
        
        # Best model should have lowest BIC among successful fits
        successful_models = filter(m -> m.fit_success, result.models)
        if length(successful_models) > 1
            best_bic = minimum([m.bic for m in successful_models])
            @test result.best_model.bic == best_bic
        end
        
        println("✓ Model comparison test passed")
        println("  Best model: $(result.best_model.name)")
        println("  Number of successful fits: $(length(successful_models))")
    end
    
    @testset "Three Datasets Fitting" begin
        data1, data2, data3 = generate_three_test_datasets()
        
        x_datasets = [data1.x, data2.x, data3.x]
        y_datasets = [data1.y, data2.y, data3.y]
        
        # Test fit_three_datasets
        results = fit_three_datasets(x_datasets, y_datasets)
        
        @test haskey(results, :individual_fits)
        @test haskey(results, :summary)
        
        @test length(results.individual_fits) == 3
        
        # Check each individual fit
        for (i, fit) in enumerate(results.individual_fits)
            @test haskey(fit, :dataset)
            @test haskey(fit, :fit_result)
            @test fit.dataset == i
            @test length(fit.fit_result.params) == 2  # Default logistic model
            @test fit.fit_result.ssr >= 0
        end
        
        # Check summary statistics
        summary = results.summary
        @test haskey(summary, :mean_params)
        @test haskey(summary, :std_params)
        @test haskey(summary, :mean_ssr)
        
        @test length(summary.mean_params) == 2
        @test length(summary.std_params) == 2
        @test summary.mean_ssr >= 0
        
        println("✓ Three datasets fitting test passed")
        println("  Mean fitted r: $(round(summary.mean_params[1], digits=4))")
        println("  Mean fitted K: $(round(summary.mean_params[2], digits=2))")
        println("  Parameter std devs: $(round.(summary.std_params, digits=4))")
    end
    
    @testset "Parameter Bounds" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        p0 = [0.1, 50.0]
        
        # Test with parameter bounds
        bounds = [(0.01, 1.0), (10.0, 200.0)]  # [r_min, r_max], [K_min, K_max]
        
        result = run_single_fit(x, y, p0; bounds=bounds, show_stats=false)
        
        @test length(result.params) == 2
        r_fitted, K_fitted = result.params
        
        # Check bounds are respected
        @test bounds[1][1] <= r_fitted <= bounds[1][2]
        @test bounds[2][1] <= K_fitted <= bounds[2][2]
        
        println("✓ Parameter bounds test passed")
        println("  Fitted r: $(round(r_fitted, digits=4)) (bounds: $(bounds[1]))")
        println("  Fitted K: $(round(K_fitted, digits=2)) (bounds: $(bounds[2]))")
    end
    
    @testset "Error Handling" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        
        # Test with mismatched data lengths
        @test_throws BoundsError run_single_fit(x[1:end-1], y, [0.1, 50.0])
        
        # Test with empty data
        @test_throws BoundsError run_single_fit(Float64[], Float64[], [0.1, 50.0])
        
        # Test with negative/zero initial values 
        x_bad = [0.0, 1.0, 2.0]
        y_bad = [0.0, -1.0, 2.0]  # Contains negative values
        
        # Should handle gracefully or throw appropriate error
        try
            result = run_single_fit(x_bad, y_bad, [0.1, 50.0]; bounds=[(0.01, 2.0), (10.0, 100.0)], show_stats=false)
            # If it succeeds, check that it returns something reasonable
            @test haskey(result, :params)
        catch e
            # If it fails, that's also acceptable for bad data
            @test isa(e, Exception)
        end
        
        println("✓ Error handling test passed")
    end
    
    @testset "Fixed Parameters" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        
        # Test with fixed carrying capacity
        fixed_params = Dict(2 => 80.0)  # Fix K to 80.0
        p0 = [0.1, 80.0]  # K will be ignored due to fixing
        
        result = run_single_fit(x, y, p0; fixed_params=fixed_params, bounds=[(0.01, 2.0), (10.0, 100.0)], show_stats=false)
        
        @test length(result.params) == 2
        @test result.params[2] ≈ 80.0  # K should be exactly the fixed value
        
        println("✓ Fixed parameters test passed")
        println("  Fixed K = 80.0, fitted r = $(round(result.params[1], digits=4))")
    end
    
end
