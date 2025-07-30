using Pkg
Pkg.activate(".")
include("src/GrowthParamEst.jl")
using .GrowthParamEst
println("✅ Package loaded successfully!")

# Test a basic function
println("Available functions: ", names(GrowthParamEst))
