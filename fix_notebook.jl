using Pkg
Pkg.add("JSON")
using JSON
infile = "C:/Users/elbak/OneDrive/Desktop/Research/Research Package/GrowthParameterEstimation.jl/test/function_tour.ipynb"
data = JSON.parse(read(infile, String))
open(infile, "w") do io
    JSON.print(io, data, 2)
end
println("Notebook fixed.")
