using V1SimpleODE
using Test
using DifferentialEquations
using Statistics
using Random

# Set a global random seed for reproducible tests
Random.seed!(12345)

println("Starting V1SimpleODE test suite...")
println("Testing package version: V1SimpleODE")

@testset "V1SimpleODE.jl Test Suite" begin
    
    @testset "Package Loading and Basic Setup" begin
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
        
        println("✓ Package structure validation passed")
    end
    
    @testset "ODE Models Validation" begin
        # Test all ODE model functions exist and work
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
        p_gompertz = [0.1, 1.0, 100.0]
        gompertz_growth!(du_test, u_test, p_gompertz, t_test)
        @test length(du_test) == 1
        @test isfinite(du_test[1])
        
        # Test exponential with delay
        p_exp_delay = [0.2, 100.0, 0.5]
        exponential_growth_with_delay!(du_test, u_test, p_exp_delay, t_test)
        @test length(du_test) == 1
        @test isfinite(du_test[1])
        
        # Test exponential with death and delay
        p_exp_death_delay = [0.2, 100.0, 0.01, 0.5]
        exponential_growth_with_death_and_delay!(du_test, u_test, p_exp_death_delay, t_test)
        @test length(du_test) == 1
        @test isfinite(du_test[1])
        
        # Test pure exponential
        p_exp = [0.2]
        exponential_growth!(du_test, u_test, p_exp, t_test)
        @test length(du_test) == 1
        @test isfinite(du_test[1])
        @test du_test[1] > 0  # Should be positive growth
        
        println("✓ All ODE models validation passed")
    end
    
    # Include the specific test files
    println("\n" * "="^50)
    println("Running Fitting Tests...")
    println("="^50)
    include("test_fitting.jl")
    
    println("\n" * "="^50)
    println("Running Analysis Tests...")  
    println("="^50)
    include("test_analysis.jl")
    
end

println("\n" * "="^60)
println("V1SimpleODE test suite completed!")
println("="^60)
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