path = ARGS[1]
content = read(path, String)
old_cell = \"\"\"# Fitting + comparisons
fit = run_single_fit(x, y, p0; model=GrowthParameterEstimation.Models.to_ode!(GrowthParameterEstimation.Models.build_logistic()), solver=Tsit5(), bounds=bounds, show_stats=false)
comp = compare_models(x, y, \\\"Logistic\\\", GrowthParameterEstimation.Models.to_ode!(GrowthParameterEstimation.Models.build_logistic()), [0.2, 12.0], \\\"Gompertz\\\", GrowthParameterEstimation.Models.to_ode!(GrowthParameterEstimation.Models.build_gompertz()), [0.2, 1.2, 12.0]; solver=Tsit5(), bounds1=bounds, bounds2=[(0.01, 2.0), (0.1, 5.0), (2.0, 100.0)], show_stats=false, output_csv=joinpath(@__DIR__, \\\"..\\\", \\\"outputs\\\", \\\"function_tour\\\", \\\"nb_compare.csv\\\"))

_ = compare_models_dict(x, y, specs; default_solver=Tsit5(), show_stats=false, output_csv=joinpath(@__DIR__, \\\"..\\\", \\\"outputs\\\", \\\"function_tour\\\", \\\"nb_compare_dict.csv\\\"))
_ = compare_models_dict(x, y, specs; default_solver=Tsit5(), show_stats=false, output_csv=joinpath(@__DIR__, \\\"..\\\", \\\"outputs\\\", \\\"function_tour\\\", \\\"nb_compare.csv\\\"))
println(\\\"Fit params: \\\", fit.params, \\\" | Best model: \\\", comp.best_model.name)\"\"\"
new_cell = \"\"\"# Fitting + comparisons
fit = run_single_fit(x, y, p0; model=GrowthParameterEstimation.Models.to_ode!(GrowthParameterEstimation.Models.build_logistic()), solver=Tsit5(), bounds=bounds, show_stats=false)
output_dir = joinpath(@__DIR__, \\\"..\\\", \\\"outputs\\\", \\\"function_tour\\\")

comp = compare_models(x, y, \\\"Logistic\\\", GrowthParameterEstimation.Models.to_ode!(GrowthParameterEstimation.Models.build_logistic()), [0.2, 12.0], \\\"Gompertz\\\", GrowthParameterEstimation.Models.to_ode!(GrowthParameterEstimation.Models.build_gompertz()), [0.2, 1.2, 12.0]; solver=Tsit5(), bounds1=bounds, bounds2=[(0.01, 2.0), (0.1, 5.0), (2.0, 100.0)], show_stats=false, output_csv=joinpath(output_dir, \\\"nb_compare.csv\\\"))

_ = compare_models_dict(x, y, specs; default_solver=Tsit5(), show_stats=false, output_csv=joinpath(output_dir, \\\"nb_compare_dict.csv\\\"))
_ = compare_models_dict(x, y, specs; default_solver=Tsit5(), show_stats=false, output_csv=joinpath(output_dir, \\\"nb_compare.csv\\\"))
println(\\\"Fit params: \\\", fit.params, \\\" | Best model: \\\", comp.best_model.name)\"\"\"
new_content = replace(content, old_cell => new_cell)
open(path, \"w\") do f
    write(f, new_content)
end