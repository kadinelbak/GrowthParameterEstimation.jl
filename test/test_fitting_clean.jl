# Focused Fitting Tests for GrowthParamEst
# Tests specific to the Fitting module functionality

using Test
using GrowthParamEst
using DifferentialEquations

include("test_data_generator.jl")

@testset "Fitting Module Tests" begin
    
    @testset "run_single_fit Function" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        p0 = [0.1, 50.0]
        bounds = [(0.01, 2.0), (10.0, 100.0)]
        
        # Test basic functionality
        result = run_single_fit(x, y, p0; bounds=bounds, show_stats=false)
        
        @test haskey(result, :params)
        @test haskey(result, :ssr)
        @test haskey(result, :bic)
        @test haskey(result, :solution)
        
        @test length(result.params) == 2
        @test result.ssr >= 0
        @test isfinite(result.bic)
        @test all(isfinite.(result.params))
        
        # Test with different models
        result_gompertz = run_single_fit(x, y, [0.1, 1.0, 50.0]; 
                                        model=gompertz_growth!, 
                                        bounds=[(0.01, 2.0), (0.1, 10.0), (10.0, 100.0)], 
                                        show_stats=false)
        @test length(result_gompertz.params) == 3
        @test result_gompertz.ssr >= 0
        
        println("✓ run_single_fit tests passed")
    end
    
    @testset "compare_models Function" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        
        # Test two-model comparison
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
        
        # Test that both models were fitted
        @test isfinite(result.model1.bic)
        @test isfinite(result.model2.bic)
        @test result.model1.ssr >= 0
        @test result.model2.ssr >= 0
        
        println("✓ compare_models tests passed")
    end
    
    @testset "fit_three_datasets Function" begin
        # Generate three different test datasets
        data1 = get_basic_test_data()
        data2 = generate_logistic_test_data(r=0.3, K=80.0, seed=123)
        data3 = generate_logistic_test_data(r=0.4, K=120.0, seed=456)
        
        x_datasets = [data1.x, data2.x, data3.x]
        y_datasets = [data1.y, data2.y, data3.y]
        bounds = [(0.01, 2.0), (10.0, 150.0)]
        
        result = fit_three_datasets(x_datasets, y_datasets; bounds=bounds, show_stats=false)
        
        @test haskey(result, :dataset1)
        @test haskey(result, :dataset2)
        @test haskey(result, :dataset3)
        @test haskey(result, :summary)
        
        # Check that all fits succeeded
        @test result.dataset1.ssr >= 0
        @test result.dataset2.ssr >= 0
        @test result.dataset3.ssr >= 0
        
        @test isfinite(result.dataset1.bic)
        @test isfinite(result.dataset2.bic)
        @test isfinite(result.dataset3.bic)
        
        println("✓ fit_three_datasets tests passed")
    end
    
    @testset "Fitting with Different Solvers" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        p0 = [0.1, 50.0]
        bounds = [(0.01, 2.0), (10.0, 100.0)]
        
        # Test with different ODE solvers
        solvers = [Rodas5(), Tsit5(), AutoTsit5(Rosenbrock23())]
        
        for solver in solvers
            result = run_single_fit(x, y, p0; bounds=bounds, solver=solver, show_stats=false)
            @test haskey(result, :params)
            @test length(result.params) == 2
            @test result.ssr >= 0
            @test isfinite(result.bic)
        end
        
        println("✓ Different solvers tests passed")
    end
    
    @testset "Error Handling" begin
        data = get_basic_test_data()
        x, y = data.x, data.y
        
        # Test with invalid bounds (should handle gracefully)
        @test_throws Exception run_single_fit(x, y, [0.1, 50.0]; bounds=[(-1.0, 0.0), (10.0, 100.0)])
        
        # Test with empty data (should handle gracefully)
        @test_throws Exception run_single_fit(Float64[], Float64[], [0.1, 50.0])
        
        # Test with mismatched x, y lengths
        @test_throws Exception run_single_fit([1.0, 2.0], [1.0], [0.1, 50.0])
        
        println("✓ Error handling tests passed")
    end
    
end
