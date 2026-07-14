
path = ARGS[1]
content = read(path, String)
# Replace the two output_csv lines
content = replace(content, "output_csv=joinpath(@__DIR__, \"..\", \"outputs\", \"function_tour\", \"nb_compare.csv\")", "output_csv=joinpath(output_dir, \"nb_compare.csv\")")
content = replace(content, "output_csv=joinpath(@__DIR__, \"..\", \"outputs\", \"function_tour\", \"nb_compare_dict.csv\")", "output_csv=joinpath(output_dir, \"nb_compare_dict.csv\")")
# Now, we want to make sure the output_dir line is present after the fit line.
lines = split(content, "\\n")
newLines = []
foundFit = false
outputDirAdded = false
for line in lines
    push!(newLines, line)
    if occursin("fit = run_single_fit(x, y, p0; model=GrowthParameterEstimation.Models.to_ode!(GrowthParameterEstimation.Models.build_logistic())", line)
        foundFit = true
    end
    if foundFit && !outputDirAdded && occursin("comp = compare_models(x, y, \"Logistic\", GrowthParameterEstimation.Models.to_ode!(GrowthParameterEstimation.Models.build_logistic())", line)
        # Insert the output_dir line before this line
        insert!(newLines, length(newLines), "output_dir = joinpath(@__DIR__, \"..\", \"outputs\", \"function_tour\")")
        outputDirAdded = true
    end
end
content = join(newLines, "\\n")
open(path, "w") do io
    write(io, content)
end
