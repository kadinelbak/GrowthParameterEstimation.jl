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
    @test sol.u[end][1] > 0  # check scalar state value, not the whole vector

    # Smoke test: evaluate BIC on a known solution
    x = [0.0, 1.0, 2.0, 3.0]
    y = [1.0, 1.8, 2.6, 3.4]
    prob_fit = ODEProblem(GrowthParameterEstimation.Models.logistic_growth!, [y[1]], (x[1], x[end]), [0.1, 5.0])
    bic, ssr = GrowthParameterEstimation.Fitting.calculate_bic(prob_fit, x, y, Tsit5(), [0.1, 5.0])
    @test ssr >= 0
    @test isfinite(bic)

    # Smoke test fitting routine (ensures run_single_fit works end-to-end)
    fit = GrowthParameterEstimation.run_single_fit(x, y, [0.1, 5.0]; solver = Tsit5(), max_time = 1.0, show_stats = false)
    @test length(fit.params) == 2
    @test isfinite(fit.bic)
end
