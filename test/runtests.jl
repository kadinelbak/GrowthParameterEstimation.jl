using Test
using DifferentialEquations
using GrowthParameterEstimation
using DataFrames

@testset "GrowthParameterEstimation" begin
    @test isdefined(GrowthParameterEstimation, :Models)

    # Basic solve of logistic growth
    u0 = [1.0]
    p  = (0.5, 10.0)  # tuple avoids param-scalar inference
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

    @testset "Data Schema Validation" begin
        raw = DataFrame(time = [0.0, 1.0, 2.0], count = [1.0, 2.0, 3.1], dose = [0.0, 0.0, 0.0], cell_line = ["A", "A", "A"], density = [1.0, 1.0, 1.0], replicate = [1, 1, 1])
        norm = normalize_schema(raw)
        @test validate_timeseries(norm)
        @test all(c -> c in Symbol.(names(norm)), REQUIRED_COLUMNS)
    end

    @testset "Registry + Simulation" begin
        models = list_models()
        @test "logistic_growth" in models

        spec = get_model("theta_logistic_hill_inhibition")
        times = collect(0.0:1.0:5.0)
        params = [0.4, 50.0, 1.0, 0.5, 1.5]
        sim = simulate(spec, times, params; u0 = [1.0], exposure = ConstantExposure(0.2))
        @test sim.success
        @test length(sim.observed) == length(times)
        @test all(isfinite, sim.observed)
    end

    @testset "Pipeline Fit + Ranking" begin
        df = DataFrame(
            time = vcat(collect(0.0:1.0:6.0), collect(0.0:1.0:6.0)),
            count = vcat([1.0, 1.5, 2.2, 3.0, 3.8, 4.5, 5.1], [1.0, 1.3, 1.9, 2.5, 3.1, 3.6, 4.0]),
            error = fill(0.2, 14),
            dose = vcat(fill(0.2, 7), fill(0.6, 7)),
            cell_line = fill("A549", 14),
            density = fill(1.0, 14),
            replicate = vcat(fill(1, 7), fill(2, 7)),
            unit_time = fill("h", 14),
            unit_count = fill("count", 14),
        )

        conditions = build_conditions(df)
        ranked = rank_models(["logistic_growth", "theta_logistic_hill_inhibition"], conditions; n_starts = 3, maxiters = 80, top_k = 2, seed = 7)

        @test nrow(ranked.ranking) == 2
        @test ranked.ranking.bic[1] <= ranked.ranking.bic[2]
        finite_deltas = filter(isfinite, ranked.ranking.delta_bic)
        @test all(finite_deltas .>= 0)
    end
end
