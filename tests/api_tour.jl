using GrowthParameterEstimation
using DataFrames
using CSV
using OrdinaryDiffEq
using DifferentialEquations
using Random

Random.seed!(42)

println("="^72)
println("GrowthParameterEstimation API Tour")
println("="^72)

x = collect(0.0:1.0:8.0)
y = [1.0, 1.6, 2.4, 3.1, 3.9, 4.6, 5.2, 5.7, 6.0]
p0 = [0.2, 12.0]

println("\n[1/8] Data layer")
raw = DataFrame(
    time = x,
    count = y,
    dose = fill(0.2, length(x)),
    cell_line = fill("A549", length(x)),
    density = fill(1.0, length(x)),
    replicate = fill(1, length(x)),
)
norm = normalize_schema(raw)
@assert validate_timeseries(norm)
println("- REQUIRED_COLUMNS: ", REQUIRED_COLUMNS)

tmp_csv = joinpath(tempdir(), "gpe_api_tour_timeseries.csv")
CSV.write(tmp_csv, norm)
loaded = load_timeseries(tmp_csv)
println("- Loaded rows: ", nrow(loaded))

println("\n[2/8] Exposure layer")
exp_constant = build_exposure(:constant; value = 0.25)
exp_pulse = build_exposure(:pulse; amplitude = 1.0, start_time = 2.0, end_time = 4.0)
exp_step = build_exposure(:stepped; change_times = [0.0, 3.0, 6.0], values = [0.0, 0.5, 0.1])
exp_decay = build_exposure(:decay; c0 = 1.0, decay_rate = 0.4, t0 = 1.0)
println("- Constant exposure at t=2: ", evaluate_exposure(exp_constant, [2.0])[1])
println("- Pulse profile: ", evaluate_exposure(exp_pulse, x))
println("- Step profile: ", evaluate_exposure(exp_step, x))
println("- Decay profile: ", round.(evaluate_exposure(exp_decay, x), digits = 3))

println("\n[3/8] Registry + simulation")
all_models = list_models()
println("- Registered models: ", length(all_models))
println("- First few models: ", first(all_models, min(5, length(all_models))))

spec = get_model("logistic_growth")
sim = simulate(spec, x, [0.4, 20.0]; u0 = [y[1]], exposure = ConstantExposure(0.2))
println("- Sim success: ", sim.success, " | observed points: ", length(sim.observed))

custom_spec = ModelSpec(
    "tour_logistic_copy",
    (du, u, p, t, exposure) -> logistic_growth!(du, u, p, t),
    [:r, :K],
    [(1e-6, 3.0), (1.0, 1e4)],
    [:N],
    (u, p, t) -> u[1],
    :ode,
    Dict(:tag => :tour),
)
register_model(custom_spec; overwrite = true)
println("- Custom model registered: ", "tour_logistic_copy" in list_models())

println("\n[4/8] Observation helpers")
obs_spec = ObservationSpec("viable_scaled", viable_total, 1.2, 0.1)
state = [3.0, 2.0, 1.0]
println("- observed_signal: ", observed_signal(obs_spec, state, nothing, 0.0))
sum_fn = sum_states([1, 2])
println("- sum_states([1,2]): ", sum_fn(state, nothing, 0.0))

println("\n[5/8] Fitting helpers")
bounds = [(0.01, 2.0), (2.0, 100.0)]
prob = ODEProblem(logistic_growth!, [y[1]], (x[1], x[end]), p0)
bic0, ssr0 = calculate_bic(prob, x, y, Tsit5(), p0)
println("- Baseline BIC/SSR: ", round(bic0, digits = 3), " / ", round(ssr0, digits = 3))

p_opt, sol_opt, prob_opt = setUpProblem(logistic_growth!, x, y, Tsit5(), [y[1]], p0, (x[1], x[end]), bounds; maxiters = 80)
fit = run_single_fit(x, y, p0; model = logistic_growth!, solver = Tsit5(), bounds = bounds, show_stats = false)
println("- run_single_fit params: ", round.(fit.params, digits = 4))
pQuickStat(x, y, fit.params, fit.solution, prob_opt, fit.bic, fit.ssr)

println("\n[6/8] Comparison + analysis")
comp = compare_models(
    x, y,
    "Logistic", logistic_growth!, [0.2, 12.0],
    "Gompertz", gompertz_growth!, [0.2, 1.2, 12.0];
    solver = Tsit5(),
    bounds1 = [(0.01, 2.0), (2.0, 100.0)],
    bounds2 = [(0.01, 2.0), (0.1, 5.0), (2.0, 100.0)],
    show_stats = false,
    output_csv = joinpath(tempdir(), "gpe_compare_models.csv"),
)
println("- Best model: ", comp.best_model.name)

compare_datasets(
    x, y, "DatasetA", logistic_growth!, [0.2, 12.0],
    x, y .* 0.95, "DatasetB", logistic_growth!, [0.2, 12.0];
    solver = Tsit5(),
    bounds1 = [(0.01, 2.0), (2.0, 100.0)],
    bounds2 = [(0.01, 2.0), (2.0, 100.0)],
    show_stats = false,
    output_csv = joinpath(tempdir(), "gpe_compare_datasets.csv"),
)

specs = Dict(
    "Logistic" => (model = logistic_growth!, p0 = [0.2, 12.0], bounds = [(0.01, 2.0), (2.0, 100.0)]),
    "Exp" => (model = exponential_growth!, p0 = [0.2], bounds = [(0.001, 2.0)]),
)
compare_models_dict(x, y, specs; default_solver = Tsit5(), show_stats = false, output_csv = joinpath(tempdir(), "gpe_compare_dict.csv"))

three = fit_three_datasets(
    x, y, "A",
    x, y .* 0.9, "B",
    x, y .* 1.1, "C",
    [0.2, 12.0];
    model = logistic_growth!,
    solver = Tsit5(),
    bounds = [(0.01, 2.0), (2.0, 100.0)],
    show_stats = false,
    output_csv = joinpath(tempdir(), "gpe_three.csv"),
)
println("- fit_three_datasets BICs: ", (three.fit1.bic, three.fit2.bic, three.fit3.bic))

loo = leave_one_out_validation(x, y, [0.2, 12.0]; model = logistic_growth!, solver = Tsit5(), bounds = bounds, show_stats = false)
kfold = k_fold_cross_validation(x, y, [0.2, 12.0]; k_folds = 3, model = logistic_growth!, solver = Tsit5(), bounds = bounds, show_stats = false)
sens = parameter_sensitivity_analysis(x, y, fit; perturbation = 0.1, model = logistic_growth!, solver = Tsit5())
resid = residual_analysis(x, y, fit; model = logistic_growth!, solver = Tsit5(), outlier_threshold = 2.0)
enh = enhanced_bic_analysis(x, y; models = [logistic_growth!, gompertz_growth!], model_names = ["Logistic", "Gompertz"], p0_values = [[0.2, 12.0], [0.2, 1.2, 12.0]], solver = Tsit5())
println("- LOO RMSE: ", round(loo.rmse, digits = 4), " | KFold RMSE: ", round(kfold.overall_rmse, digits = 4))
println("- Sensitivity ranked count: ", length(sens.ranking), " | Residual RMSE: ", round(resid.statistics.rmse, digits = 4))
println("- Enhanced BIC best: ", enh.best_model.model_name)

println("\n[7/8] Workflow APIs")
df_work = DataFrame(
    time = vcat(x, x),
    count = vcat(y, y .* 0.9),
    error = fill(0.2, 2 * length(x)),
    dose = vcat(fill(0.2, length(x)), fill(0.6, length(x))),
    cell_line = fill("A549", 2 * length(x)),
    density = fill(1.0, 2 * length(x)),
    replicate = vcat(fill(1, length(x)), fill(2, length(x))),
    unit_time = fill("h", 2 * length(x)),
    unit_count = fill("count", 2 * length(x)),
)

conds = build_conditions(df_work)
ranked = rank_models(["logistic_growth", "theta_logistic_hill_inhibition"], conds; n_starts = 2, maxiters = 30, top_k = 2, seed = 42)
exports = export_results(ranked; output_dir = joinpath(tempdir(), "gpe_api_tour_exports"))
println("- Conditions: ", length(conds), " | Ranking rows: ", nrow(ranked.ranking))
println("- Exported ranking file: ", exports.ranking)

cfg = default_config(output_dir = joinpath(tempdir(), "gpe_api_tour_pipeline"))
cfg_path = save_config(joinpath(tempdir(), "gpe_config.toml"), cfg)
cfg_loaded = load_config(cfg_path)
pipe = run_pipeline(df_work; config = cfg_loaded, include_models = ["logistic_growth", "gompertz_growth"])
println("- Pipeline rows: ", nrow(pipe.ranking), " | Pipeline plots generated: ", length(pipe.plots))

println("\n[8/8] Done")
println("API tour completed successfully.")
println("="^72)
