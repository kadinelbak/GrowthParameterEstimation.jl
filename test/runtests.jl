using Test
using DifferentialEquations
using GrowthParameterEstimation
using DataFrames
using CSV
using Random

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

        csv_path = joinpath(tempdir(), "gpe_test_timeseries.csv")
        CSV.write(csv_path, norm)
        loaded = load_timeseries(csv_path)
        @test nrow(loaded) == nrow(norm)
    end

    @testset "Exposure + Observation" begin
        times = collect(0.0:1.0:5.0)
        exp_constant = build_exposure(:constant; value = 0.25)
        exp_pulse = build_exposure(:pulse; amplitude = 1.0, start_time = 2.0, end_time = 4.0)
        exp_step = build_exposure(:stepped; change_times = [0.0, 3.0], values = [0.1, 0.4])
        exp_decay = build_exposure(:decay; c0 = 1.0, decay_rate = 0.3, t0 = 1.0)

        @test length(evaluate_exposure(exp_constant, times)) == length(times)
        @test evaluate_exposure(exp_pulse, [1.0, 3.0]) == [0.0, 1.0]
        @test length(evaluate_exposure(exp_step, times)) == length(times)
        @test length(evaluate_exposure(exp_decay, times)) == length(times)

        obs = ObservationSpec("viable_scaled", viable_total, 1.2, 0.1)
        state = [3.0, 2.0]
        @test observed_signal(obs, state, nothing, 0.0) ≈ 3.6
        sum_fn = sum_states([1, 2])
        @test sum_fn(state, nothing, 0.0) == 5.0
    end

    @testset "Model RHS Smoke" begin
        du = [0.0]
        logistic_growth!(du, [2.0], [0.2, 10.0], 0.0)
        @test isfinite(du[1])

        logistic_growth_with_death!(du, [2.0], [0.2, 10.0, 0.01], 0.0)
        @test isfinite(du[1])

        gompertz_growth!(du, [2.0], [0.2, 1.0, 10.0], 0.0)
        @test isfinite(du[1])

        gompertz_growth_with_death!(du, [2.0], [0.2, 1.0, 10.0, 0.01], 0.0)
        @test isfinite(du[1])

        exponential_growth!(du, [2.0], [0.2], 0.0)
        @test isfinite(du[1])

        exponential_growth_with_delay!(du, [2.0], [0.2, 10.0, 1.0], 2.0)
        @test isfinite(du[1])

        logistic_growth_with_delay!(du, [2.0], [0.2, 10.0, 1.0], 2.0)
        @test isfinite(du[1])

        exponential_growth_with_death_and_delay!(du, [2.0], [0.2, 10.0, 0.01, 1.0], 2.0)
        @test isfinite(du[1])
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

        # Workflow config + plotting + export + pipeline run
        cfg = default_config(output_dir = joinpath(tempdir(), "gpe_pipeline_test"))
        cfg_path = save_config(joinpath(tempdir(), "gpe_cfg_test.toml"), cfg)
        cfg_loaded = load_config(cfg_path)
        @test cfg_loaded.top_k == cfg.top_k

        # Use a single-condition subset for deterministic export behavior
        df_one = df[df.replicate .== 1, :]
        conditions_one = build_conditions(df_one)
        ranked_logistic = rank_models(["logistic_growth"], conditions_one; n_starts = 3, maxiters = 80, top_k = 1, seed = 11)
        generated = plot_topk(ranked_logistic; conditions = conditions_one, top_k = 1, output_dir = joinpath(tempdir(), "gpe_plot_topk_test"))
        @test generated isa Vector{String}

        try
            exports = export_results(ranked_logistic; output_dir = joinpath(tempdir(), "gpe_export_test"))
            @test isfile(exports.ranking)
            @test isfile(exports.params)
            @test isfile(exports.failures)
        catch err
            @test occursin("No successful model fits available to export", string(err))
        end

        pipe = run_pipeline(df; config = cfg_loaded, include_models = ["logistic_growth"])
        @test nrow(pipe.ranking) == 1
    end

    @testset "Fitting Compare APIs" begin
        x = collect(0.0:1.0:6.0)
        y = [1.0, 1.5, 2.1, 2.9, 3.7, 4.3, 4.9]

        comp = compare_models(
            x, y,
            "Logistic", logistic_growth!, [0.2, 8.0],
            "Gompertz", gompertz_growth!, [0.2, 1.0, 8.0];
            solver = Tsit5(),
            bounds1 = [(0.01, 2.0), (2.0, 100.0)],
            bounds2 = [(0.01, 2.0), (0.1, 5.0), (2.0, 100.0)],
            show_stats = false,
            output_csv = joinpath(tempdir(), "gpe_compare_models_test.csv"),
        )
        @test haskey(comp, :best_model)

        compare_datasets(
            x, y, "A", logistic_growth!, [0.2, 8.0],
            x, y .* 0.95, "B", logistic_growth!, [0.2, 8.0];
            solver = Tsit5(),
            bounds1 = [(0.01, 2.0), (2.0, 100.0)],
            bounds2 = [(0.01, 2.0), (2.0, 100.0)],
            show_stats = false,
            output_csv = joinpath(tempdir(), "gpe_compare_datasets_test.csv"),
        )
        @test isfile(joinpath(tempdir(), "gpe_compare_datasets_test.csv"))

        specs = Dict(
            "Logistic" => (model = logistic_growth!, p0 = [0.2, 8.0], bounds = [(0.01, 2.0), (2.0, 100.0)]),
            "Exp" => (model = exponential_growth!, p0 = [0.2], bounds = [(0.001, 2.0)]),
        )
        dict_fits = compare_models_dict(x, y, specs; default_solver = Tsit5(), show_stats = false, output_csv = joinpath(tempdir(), "gpe_compare_dict_test.csv"))
        @test haskey(dict_fits, "Logistic")

        three_named = fit_three_datasets(
            x, y, "A",
            x, y .* 0.9, "B",
            x, y .* 1.1, "C",
            [0.2, 8.0];
            model = logistic_growth!,
            solver = Tsit5(),
            bounds = [(0.01, 2.0), (2.0, 100.0)],
            show_stats = false,
            output_csv = joinpath(tempdir(), "gpe_three_named_test.csv"),
        )
        @test haskey(three_named, :fit1)

        x_many = [collect(x), collect(x)]
        y_many = [collect(y), collect(y .* 0.95)]
        three_many = fit_three_datasets(x_many, y_many; model = logistic_growth!, solver = Tsit5(), bounds = [(0.01, 2.0), (2.0, 100.0)])
        @test three_many.summary.n_total == 2
    end

    @testset "Analysis APIs" begin
        Random.seed!(7)
        x = collect(0.0:1.0:8.0)
        y = [1.0, 1.6, 2.4, 3.1, 3.9, 4.6, 5.2, 5.7, 6.0]
        p0 = [0.2, 12.0]
        bounds = [(0.01, 2.0), (2.0, 100.0)]

        fit = run_single_fit(x, y, p0; model = logistic_growth!, solver = Tsit5(), bounds = bounds, show_stats = false)

        loo = leave_one_out_validation(x, y, p0; model = logistic_growth!, solver = Tsit5(), bounds = bounds, show_stats = false)
        @test isfinite(loo.rmse)

        kfold = k_fold_cross_validation(x, y, p0; k_folds = 3, model = logistic_growth!, solver = Tsit5(), bounds = bounds, show_stats = false)
        @test isfinite(kfold.overall_rmse)

        sens = parameter_sensitivity_analysis(x, y, fit; perturbation = 0.1, model = logistic_growth!, solver = Tsit5())
        @test length(sens.ranking) >= 1

        resid = residual_analysis(x, y, fit; model = logistic_growth!, solver = Tsit5())
        @test isfinite(resid.statistics.rmse)

        enh = enhanced_bic_analysis(
            x,
            y;
            models = [logistic_growth!],
            model_names = ["Logistic"],
            p0_values = [[0.2, 12.0]],
            solver = Tsit5(),
        )
        @test enh.best_model.model_name == "Logistic"
    end

    @testset "Joint Fitting APIs" begin
        function logistic_joint!(du, u, p, t)
            r, K = p
            du[1] = r * u[1] * (1 - u[1] / K)
            du[2] = r * u[2] * (1 - u[2] / K)
        end

        x1 = collect(0.0:1.0:5.0)
        x2 = collect(0.0:1.0:5.0)
        u0_joint = [1.0, 2.0]
        p_true = [0.35, 25.0]
        prob_joint = ODEProblem(logistic_joint!, u0_joint, (x1[1], x1[end]), p_true)
        sol_joint = solve(prob_joint, Tsit5(); saveat = x1)
        y1 = [u[1] for u in sol_joint.u]
        y2 = [u[2] for u in sol_joint.u]

        datasets = [
            (x = x1, y = y1, state_index = 1),
            (x = x2, y = y2, state_index = 2),
        ]

        joint_fit = run_joint_fit(logistic_joint!, datasets, u0_joint, [0.2, 20.0]; bounds = [(0.01, 1.0), (5.0, 60.0)], show_stats = false)
        @test length(joint_fit.params) == 2
        @test isfinite(joint_fit.bic)
        @test joint_fit.sse >= 0

        specs = Dict(
            "LogisticJoint" => (model = logistic_joint!, p0 = [0.2, 20.0], bounds = [(0.01, 1.0), (5.0, 60.0)]),
        )
        joint_models = compare_joint_models_dict(datasets, u0_joint, specs; default_solver = Tsit5(), show_stats = false, output_csv = joinpath(tempdir(), "joint_compare_smoke.csv"))
        @test haskey(joint_models, "LogisticJoint")
    end
end
