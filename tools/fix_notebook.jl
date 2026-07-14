
using JSON
path = ARGS[1]
# Read the notebook
open(path, "r") do f
    nb = JSON.parse(f)
end
# Find the cell with "# Fitting + comparisons"
for i in eachindex(nb["cells"])
    cell = nb["cells"][i]
    if cell["cell_type"] == "code" && length(cell["source"]) > 0 && occursin("# Fitting + comparisons", cell["source"][1])
        # Replace the source
        cell["source"] = [
            "# Fitting + comparisons\\n",
            "fit = run_single_fit(x, y, p0; model=GrowthParameterEstimation.Models.to_ode!(GrowthParameterEstimation.Models.build_logistic()), solver=Tsit5(), bounds=bounds, show_stats=false)\\n",
            "output_dir = joinpath(@__DIR__, \"..\", \"outputs\", \"function_tour\")\\n",
            "comp = compare_models(x, y, \"Logistic\", GrowthParameterEstimation.Models.to_ode!(GrowthParameterEstimation.Models.build_logistic()), [0.2, 12.0], \"Gompertz\", GrowthParameterEstimation.Models.to_ode!(GrowthParameterEstimation.Models.build_gompertz()), [0.2, 1.2, 12.0]; solver=Tsit5(), bounds1=bounds, bounds2=[(0.01, 2.0), (0.1, 5.0), (2.0, 100.0)], show_stats=false, output_csv=joinpath(output_dir, \"nb_compare.csv\"))\\n",
            "\\n",
            "_ = compare_models_dict(x, y, specs; default_solver=Tsit5(), show_stats=false, output_csv=joinpath(output_dir, \"nb_compare_dict.csv\"))\\n",
            "_ = compare_models_dict(x, y, specs; default_solver=Tsit5(), show_stats=false, output_csv=joinpath(output_dir, \"nb_compare.csv\"))\\n",
            "println(\"Fit params: \", fit.params, \" | Best model: \", comp.best_model.name)\\n"
        ]
        break
    end
end
# Write the notebook back
open(path, "w") do f
    JSON.print(f, nb, 2)
end
