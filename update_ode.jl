using Pkg
Pkg.update("OrdinaryDiffEq")
Pkg.update("StochasticDiffEq")
Pkg.update("DiffEqBase")
Pkg.update("DifferentialEquations")
Pkg.precompile()
Pkg.test()
