using V1SimpleODE
using Test
using CSV, DataFrames, DifferentialEquations

@testset "V1SimpleODE.jl" begin
    # Load data
    df = CSV.read("logistic_day_averages.csv", DataFrame)
    x, y = V1SimpleODE.extractData(df)
    @test length(x) == length(y) > 0

    # Define a simple logistic model for testing
    function logistic!(du, u, p, t)
        r, K = p
        du[1] = r * u[1] * (1 - u[1]/K)
    end
    u0 = [y[1]]
    p = [0.1, maximum(y)*1.5]
    tspan = (x[1], x[end])
    bounds = [(0.0, 2.0), (maximum(y), maximum(y)*2.0)]
    solver = Tsit5()

    # Test setUpProblem
    opt_params, opt_sol, opt_prob = V1SimpleODE.setUpProblem(
        logistic!, x, y, solver, u0, p, tspan, bounds
    )
    @test length(opt_params) == 2
    @test typeof(opt_sol.t) <: AbstractVector
    @test typeof(opt_prob) <: ODEProblem

    # Test calculate_bic
    bic, ssr = V1SimpleODE.calculate_bic(opt_prob, x, y, solver, opt_params)
    @test isa(bic, Number)
    @test isa(ssr, Number)

    # Test pQuickStat (just check it runs)
    V1SimpleODE.pQuickStat(x, y, opt_params, opt_sol, opt_prob, bic, ssr)

    # Test compareModelsBB (just check it runs)
    V1SimpleODE.compareModelsBB(
        "Logistic1", "Logistic2", logistic!, logistic!, x, y, solver, u0, p, tspan, bounds;
        output_csv="test_model_comparison.csv"
    )
end