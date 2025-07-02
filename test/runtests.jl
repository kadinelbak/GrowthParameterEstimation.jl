using V1SimpleODE
using Test
using CSV, DataFrames, DifferentialEquations
using BlackBoxOptim
using Plots

# Get the directory where the test file is located
const TEST_DIR = dirname(@__FILE__)

@testset "V1SimpleODE.jl Comprehensive Tests" begin
    
    @testset "Data Extraction Functions" begin
        # Test basic extractData function
        df_basic = CSV.read(joinpath(TEST_DIR, "test_data.csv"), DataFrame)
        @test names(df_basic) == ["Day Averages"]
        
        x, y = V1SimpleODE.extractData(df_basic)
        @test length(x) == length(y)
        @test length(x) > 0
        @test all(x .== 1:length(x))
        @test all(y .> 0)  # Assuming positive values
        
        # Test extract_day_averages_from_df with tile data
        df_tiles = CSV.read(joinpath(TEST_DIR, "tile_test_data.csv"), DataFrame)
        x_tiles, y_tiles = V1SimpleODE.extract_day_averages_from_df(df_tiles)
        @test length(x_tiles) == length(y_tiles)
        @test length(x_tiles) > 0
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
        df = CSV.read(joinpath(TEST_DIR, "test_data.csv"), DataFrame)
        x, y = V1SimpleODE.extractData(df)
        
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
        df = CSV.read(joinpath(TEST_DIR, "test_data.csv"), DataFrame)
        x, y = V1SimpleODE.extractData(df)
        
        # Test run_single_fit
        @testset "run_single_fit" begin
            result = V1SimpleODE.run_single_fit(
                "TestLogistic",
                V1SimpleODE.logistic_growth!,
                x, y,
                Tsit5(),
                [y[1]],
                [0.1, maximum(y)*1.2],
                (x[1], x[end]),
                [(0.01, 1.0), (maximum(y)*0.5, maximum(y)*2.0)]
            )
            
            @test haskey(result, :name)
            @test haskey(result, :params)
            @test haskey(result, :bic)
            @test haskey(result, :ssr)
            @test haskey(result, :solution)
            @test haskey(result, :problem)
            @test result[:name] == "TestLogistic"
            @test length(result[:params]) == 2
        end
    end
    
    @testset "Model Comparison Functions" begin
        df = CSV.read(joinpath(TEST_DIR, "test_data.csv"), DataFrame)
        x, y = V1SimpleODE.extractData(df)
        
        # Test compare_models
        @testset "compare_models" begin
            results = V1SimpleODE.compare_models(
                ["Logistic", "Gompertz"],
                [V1SimpleODE.logistic_growth!, V1SimpleODE.gompertz_growth!],
                x, y,
                Tsit5(),
                [[y[1]], [y[1]]],
                [[0.1, maximum(y)*1.2], [0.05, 2.0, maximum(y)*1.2]],
                (x[1], x[end]),
                [[(0.01, 1.0), (maximum(y)*0.5, maximum(y)*2.0)],
                 [(0.01, 0.5), (0.1, 5.0), (maximum(y)*0.5, maximum(y)*2.0)]];
                output_csv = joinpath(TEST_DIR, "test_compare_models.csv")
            )
            
            @test length(results) == 2
            @test isfile(joinpath(TEST_DIR, "test_compare_models.csv"))
        end
        
        # Test compare_datasets  
        @testset "compare_datasets" begin
            # Create second dataset (slightly modified)
            y2 = y .* 1.1  # 10% increase
            
            results = V1SimpleODE.compare_datasets(
                ["Dataset1", "Dataset2"],
                [x, x],
                [y, y2],
                "SharedLogistic",
                V1SimpleODE.logistic_growth!,
                Tsit5(),
                [y[1]],
                [0.1, maximum(y)*1.2],
                (x[1], x[end]),
                [(0.01, 1.0), (maximum(y)*0.5, maximum(y)*2.0)];
                output_csv = joinpath(TEST_DIR, "test_compare_datasets.csv")
            )
            
            @test length(results) == 2
            @test isfile(joinpath(TEST_DIR, "test_compare_datasets.csv"))
        end
    end
    
    @testset "Dictionary-based Model Comparison" begin
        df = CSV.read(joinpath(TEST_DIR, "test_data.csv"), DataFrame)
        x, y = V1SimpleODE.extractData(df)
        
        # Test compare_models_dict
        @testset "compare_models_dict" begin
            model_dict = Dict(
                "Logistic" => (
                    model = V1SimpleODE.logistic_growth!,
                    u0 = [y[1]],
                    p0 = [0.1, maximum(y)*1.2],
                    bounds = [(0.01, 1.0), (maximum(y)*0.5, maximum(y)*2.0)]
                ),
                "Gompertz" => (
                    model = V1SimpleODE.gompertz_growth!,
                    u0 = [y[1]],
                    p0 = [0.05, 2.0, maximum(y)*1.2],
                    bounds = [(0.01, 0.5), (0.1, 5.0), (maximum(y)*0.5, maximum(y)*2.0)]
                )
            )
            
            results = V1SimpleODE.compare_models_dict(
                model_dict,
                x, y,
                Tsit5(),
                (x[1], x[end]);
                output_csv = joinpath(TEST_DIR, "test_models_dict.csv")
            )
            
            @test length(results) == 2
            @test haskey(results, "Logistic")
            @test haskey(results, "Gompertz")
            @test isfile(joinpath(TEST_DIR, "test_models_dict.csv"))
        end
    end
end