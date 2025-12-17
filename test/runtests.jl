using Test
using DifferentialEquations
using GrowthParameterEstimation

@testset "GrowthParameterEstimation" begin
    @test isdefined(GrowthParameterEstimation, :Models)

    # Basic solve of logistic growth
    u0 = [1.0]
    p  = [0.5, 10.0]
    tspan = (0.0, 2.0)
    prob = ODEProblem(GrowthParameterEstimation.Models.logistic_growth!, u0, tspan, p)
    sol  = solve(prob, Tsit5(); saveat = 0.5)
    @test sol[end] > 0

    # Smoke test fitting routine
    x = [0.0, 1.0, 2.0, 3.0]
    y = [1.0, 1.8, 2.6, 3.4]
    fit = GrowthParameterEstimation.run_single_fit(x, y, [0.1, 5.0]; max_time = 1.0, show_stats = false)
    @test length(fit.params) == 2
    @test isfinite(fit.bic)
end
