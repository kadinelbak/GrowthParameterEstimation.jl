using V1SimpleODE
using Test
using CSV, DataFrames, DifferentialEquations
using BlackBoxOptim
using Plots

# Get the directory where the test file is located
const TEST_DIR = dirname(@__FILE__)

@testset "V1SimpleODE.jl Comprehensive Tests" begin
    
    @testset "Test Data Setup" begin
        # Create test data instead of loading from files
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 20.0, 35.0, 55.0, 80.0]
        @test length(x) == length(y)
        @test length(x) > 0
        @test all(y .> 0)  # Assuming positive values
        
        # Create second dataset for comparison tests
        x2 = [1.0, 2.0, 3.0, 4.0, 5.0]
        y2 = [15.0, 28.0, 45.0, 65.0, 90.0]
        @test length(x2) == length(y2)
        @test length(x2) > 0
    end
    
    @testset "ODE Models" begin
        # Test all built-in ODE models
        u_test = [100.0]
        p_logistic = [0.1, 500.0]
        p_gompertz = [0.05, 2.0, 500.0]
        p_delay = [0.1, 500.0, 1.0]
        p_death = [0.1, 500.0, 0.01]
        t_test = 1.0
        du_test = similar(u_test)
        
        # Test logistic growth
        V1SimpleODE.logistic_growth!(du_test, u_test, p_logistic, t_test)
        @test length(du_test) == 1
        @test isfinite(du_test[1])
        
        # Test logistic with death
        V1SimpleODE.logistic_growth_with_death!(du_test, u_test, p_death, t_test)
        @test length(du_test) == 1
        @test isfinite(du_test[1])
        
        # Test Gompertz growth
        V1SimpleODE.gompertz_growth!(du_test, u_test, p_gompertz, t_test)
        @test length(du_test) == 1
        @test isfinite(du_test[1])
        
        # Test Gompertz with death
        p_gompertz_death = [0.05, 2.0, 500.0, 0.01]
        V1SimpleODE.gompertz_growth_with_death!(du_test, u_test, p_gompertz_death, t_test)
        @test length(du_test) == 1
        @test isfinite(du_test[1])
        
        # Test exponential with delay
        V1SimpleODE.exponential_growth_with_delay!(du_test, u_test, p_delay, t_test)
        @test length(du_test) == 1
        @test isfinite(du_test[1])
        
        # Test logistic with delay
        V1SimpleODE.logistic_growth_with_delay!(du_test, u_test, p_delay, t_test)
        @test length(du_test) == 1
        @test isfinite(du_test[1])
    end
    
    @testset "Model Fitting and Analysis" begin
        # Use simple test data
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 20.0, 35.0, 55.0, 80.0]
        
        # Define test models
        function simple_logistic!(du, u, p, t)
            r, K = p
            du[1] = r * u[1] * (1 - u[1]/K)
        end
        
        # Setup parameters
        u0 = [y[1]]
        p0 = [0.1, maximum(y)*1.2]
        tspan = (x[1], x[end])
        bounds = [(0.01, 1.0), (maximum(y)*0.5, maximum(y)*2.0)]
        solver = Tsit5()
        
        # Test setUpProblem
        @testset "setUpProblem" begin
            opt_params, opt_sol, opt_prob = V1SimpleODE.setUpProblem(
                simple_logistic!, x, y, solver, u0, p0, tspan, bounds
            )
            @test length(opt_params) == 2
            @test typeof(opt_sol.t) <: AbstractVector
            @test typeof(opt_prob) <: ODEProblem
            @test all(isfinite.(opt_params))
        end
        
        # Test calculate_bic
        @testset "calculate_bic" begin
            opt_params, opt_sol, opt_prob = V1SimpleODE.setUpProblem(
                simple_logistic!, x, y, solver, u0, p0, tspan, bounds
            )
            bic, ssr = V1SimpleODE.calculate_bic(opt_prob, x, y, solver, opt_params)
            @test isa(bic, Number)
            @test isa(ssr, Number)
            @test isfinite(bic)
            @test isfinite(ssr)
            @test ssr >= 0
        end
        
        # Test pQuickStat (just verify it doesn't error)
        @testset "pQuickStat" begin
            opt_params, opt_sol, opt_prob = V1SimpleODE.setUpProblem(
                simple_logistic!, x, y, solver, u0, p0, tspan, bounds
            )
            bic, ssr = V1SimpleODE.calculate_bic(opt_prob, x, y, solver, opt_params)
            
            # This should not throw an error
            @test_nowarn V1SimpleODE.pQuickStat(x, y, opt_params, opt_sol, opt_prob, bic, ssr)
        end
    end
    
    @testset "Single Model Fitting" begin
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 20.0, 35.0, 55.0, 80.0]
        
        # Test run_single_fit
        @testset "run_single_fit" begin
            result = V1SimpleODE.run_single_fit(
                x, y, [0.5, 100.0];
                model = V1SimpleODE.logistic_growth!,
                show_stats = false
            )
            
            @test haskey(result, :params)
            @test haskey(result, :bic)
            @test haskey(result, :ssr)
            @test haskey(result, :sol)
            @test length(result[:params]) == 2
        end
    end
    
    @testset "Model Comparison Functions" begin
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 20.0, 35.0, 55.0, 80.0]
        
        # Test compare_models
        @testset "compare_models" begin
            V1SimpleODE.compare_models(
                x, y,
                "Logistic", V1SimpleODE.logistic_growth!, [0.5, 100.0],
                "Gompertz", V1SimpleODE.gompertz_growth!, [0.3, 0.1];
                output_csv = joinpath(TEST_DIR, "test_compare_models.csv")
            )
            
            @test isfile(joinpath(TEST_DIR, "test_compare_models.csv"))
        end
        
        # Test compare_datasets  
        @testset "compare_datasets" begin
            # Create second dataset (slightly modified)
            x2 = [1.0, 2.0, 3.0, 4.0, 5.0]
            y2 = [15.0, 28.0, 45.0, 65.0, 90.0]
            
            V1SimpleODE.compare_datasets(
                x, y, "Dataset1", V1SimpleODE.logistic_growth!, [0.5, 100.0],
                x2, y2, "Dataset2", V1SimpleODE.logistic_growth!, [0.5, 100.0];
                output_csv = joinpath(TEST_DIR, "test_compare_datasets.csv")
            )
            
            @test isfile(joinpath(TEST_DIR, "test_compare_datasets.csv"))
        end
    end
    
    @testset "Dictionary-based Model Comparison" begin
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 20.0, 35.0, 55.0, 80.0]
        
        # Test compare_models_dict
        @testset "compare_models_dict" begin
            model_dict = Dict(
                "Logistic" => (
                    model = V1SimpleODE.logistic_growth!,
                    p0 = [0.1, maximum(y)*1.2],
                    bounds = [(0.01, 1.0), (maximum(y)*0.5, maximum(y)*2.0)],
                    fixed_params = nothing
                "Gompertz" => (
                    model = V1SimpleODE.gompertz_growth!,
                    p0 = [0.05, 2.0],
                    bounds = [(0.01, 0.5), (0.1, 5.0)],
                    fixed_params = nothing
                )
            )
            
            results = V1SimpleODE.compare_models_dict(
                x, y,
                model_dict;
                output_csv = joinpath(TEST_DIR, "test_models_dict.csv")
            )
            
            @test isa(results, Dict)
            @test haskey(results, "Logistic")
            @test haskey(results, "Gompertz")
            @test isfile(joinpath(TEST_DIR, "test_models_dict.csv"))
        end
    end
end