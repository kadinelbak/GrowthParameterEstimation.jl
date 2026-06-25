module Simulation

using DifferentialEquations
using DataFrames

using ..Exposure
using ..Registry

export SimulationResult, SweepGrid, SweepResult, simulate, classify_failure, run_sweep

"""
    SimulationResult{T}

Result of a simulation run.

# Fields
- `success::Bool`: Whether the simulation completed successfully
- `reason::String`: Reason for failure or "ok" if successful
- `times::Vector{Float64}`: Time points at which the solution was saved
- `states::Matrix{T}`: State variable trajectories (rows = states, columns = time points)
- `observed::Vector{T}`: Observable output at each time point
"""
struct SimulationResult{T}
    success::Bool
    reason::String
    times::Vector{Float64}
    states::Matrix{T}
    observed::Vector{T}
end

"""
    SweepGrid

Parameter grid definition for running parameter sweeps in simulations.

# Fields
- `seed_totals::Vector{Float64}`: Initial population sizes to sweep over
- `resistant_fractions::Vector{Float64}`: Fractions of resistant population to sweep over
- `doses::Vector{Float64}`: Drug doses to sweep over
- `times::Vector{Float64}`: Time points for simulation output
"""
struct SweepGrid
    seed_totals::Vector{Float64}
    resistant_fractions::Vector{Float64}
    doses::Vector{Float64}
    times::Vector{Float64}
end

"""
    SweepResult

Results from a parameter sweep simulation.

# Fields
- `summary::DataFrame`: Summary table of sweep results with one row per parameter combination
- `simulations::Dict{NamedTuple,SimulationResult}`: Dictionary mapping parameter combinations to their full simulation results
"""
struct SweepResult
    summary::DataFrame
    simulations::Dict{NamedTuple,SimulationResult}
end

"""
    classify_failure(error_obj)

Classify the type of simulation failure based on an error message.

This function analyzes error messages from ODE solvers to categorize failures
into types that can help diagnose issues with model specification or parameter values.

# Arguments
- `error_obj`: An error object or exception from a failed simulation

# Returns
- `String`: One of:
  - `"solver_failure"`: Solver-related issues like timestep problems or instability
  - `"biological_domain_failure"`: Issues like negative values in domains that require positivity
  - `"unknown_failure"`: Any other type of failure

# Examples
```julia
# Classify a solver failure due to small timestep
err = ArgumentError("dt <= dtmin. Aborting. There is either an error in your model formulation or the true solution is slowly varying.")
failure_type = classify_failure(err)  # returns "solver_failure"

# Classify a biological domain failure due to negative values
err = DomainError(-1.0, "sqrt was called with a negative argument")
failure_type = classify_failure(err)  # returns "biological_domain_failure"
```
"""
function classify_failure(error_obj)
    msg = lowercase(string(error_obj))
    if occursin("dt", msg) || occursin("unstable", msg) || occursin("instability", msg)
        return "solver_failure"
    elseif occursin("domain", msg) || occursin("nan", msg)
        return "biological_domain_failure"
    else
        return "unknown_failure"
    end
end

"""
    _solver_from_symbol(kind::Symbol)

Convert a solver type symbol to a DifferentialEquations.jl solver instance.

# Arguments
- `kind::Symbol`: Either :ode or :stiff_ode

# Returns
- `Union{Tsit5, Rodas5}`: An appropriate solver instance
"""
function _solver_from_symbol(kind::Symbol)
    if kind == :ode
        return Tsit5()
    elseif kind == :stiff_ode
        return Rodas5()
    else
        return Tsit5()
    end
end

"""
    _to_state_matrix(sol)

Convert an ODE solution object to a state matrix.

Extracts the state trajectories from an ODE solution and organizes them into
a matrix where rows correspond to state variables and columns to time points.

# Arguments
- `sol`: An ODE solution object from DifferentialEquations.jl

# Returns
- `Matrix{T}`: State matrix with dimensions (n_states, n_timepoints)
"""
function _to_state_matrix(sol)
    n_t = length(sol.u)
    n_s = length(sol.u[1])
    arr = Matrix{eltype(sol.u[1])}(undef, n_s, n_t)
    for i in 1:n_t
        for j in 1:n_s
            arr[j, i] = sol.u[i][j]
        end
    end
    return arr
end

"""
    simulate(spec::Registry.ModelSpec, times::AbstractVector{<:Real>, params::AbstractVector{<:Real>}; 
             u0::AbstractVector{<:Real>, exposure::Exposure.AbstractExposure = Exposure.ConstantExposure(0.0),
             reltol::Float64 = 1e-8, abstol::Float64 = 1e-8, enforce_nonnegative::Bool = true)

Run a simulation with the given model specification and parameters.

This function simulates a growth model over specified time points using the
provided parameters and initial conditions. It handles exposure functions,
solver selection, and error handling for robust simulation.

# Arguments
- `spec::Registry.ModelSpec`: The model specification to simulate
- `times::AbstractVector{<:Real>`: Time points at which to save the solution
- `params::AbstractVector{<:Real>`: Parameter values for the model
- `u0::AbstractVector{<:Real}`: Initial state vector
- `exposure::Exposure.AbstractExposure = Exposure.ConstantExposure(0.0)`: Exposure function affecting model dynamics
- `reltol::Float64 = 1e-8`: Relative tolerance for the ODE solver
- `abstol::Float64 = 1e-8`: Absolute tolerance for the ODE solver
- `enforce_nonnegative::Bool = true`: Whether to enforce non-negative state variables

# Returns
- `SimulationResult`: Object containing success status, reason, time points, state trajectories, and observed values

# Examples
```julia
# Simulate a logistic growth model
spec = Registry.get_model("logistic_growth")
times = 0.0:0.1:10.0
params = [0.5, 100.0]  # growth rate, carrying capacity
u0 = [1.0]  # initial population
result = simulate(spec, times, params; u0=u0)

# Check if simulation was successful
if result.success
    # Plot or analyze results
    plot(result.times, result.states[1, :])  # population over time
end
```
"""
    t_obs = Float64.(collect(times))
    tspan = (minimum(t_obs), maximum(t_obs))
    param_vec = collect(params)
    value_type = promote_type(Float64, eltype(param_vec), eltype(u0))

    wrapped! = function (du, u, p, t)
        spec.dynamics!(du, u, p, t, exposure)
        if enforce_nonnegative
            for i in eachindex(du)
                if !isfinite(du[i])
                    du[i] = 0.0
                end
            end
        end
        return nothing
    end

    prob = ODEProblem(wrapped!, value_type.(u0), tspan, value_type.(param_vec))

    try
        solver = _solver_from_symbol(spec.solver_type)
        sol = solve(prob, solver; saveat=t_obs, reltol=reltol, abstol=abstol, isoutofdomain=(u, p, t) -> any(x -> x < -1e-10 || !isfinite(x), u))

        if sol.retcode != ReturnCode.Success
            return SimulationResult(false, "solver_failure: $(sol.retcode)", t_obs, zeros(Float64, length(u0), length(t_obs)), fill(NaN, length(t_obs)))
        end

        state_matrix = _to_state_matrix(sol)
        if enforce_nonnegative
            state_matrix .= max.(state_matrix, 0.0)
        end

        observed = [spec.observation(state_matrix[:, i], params, t_obs[i]) for i in eachindex(t_obs)]
        return SimulationResult(true, "ok", t_obs, state_matrix, observed)
    catch err
        return SimulationResult(false, classify_failure(err), t_obs, zeros(Float64, length(u0), length(t_obs)), fill(NaN, length(t_obs)))
    end
end

function _make_sweep_u0(spec::Registry.ModelSpec, seed_total::Real, resistant_fraction::Real)
    n_states = length(spec.state_names)
    u0 = zeros(Float64, n_states)
    total = max(Float64(seed_total), 0.0)
    frac_r = clamp(Float64(resistant_fraction), 0.0, 1.0)

    if n_states == 1
        u0[1] = total
    elseif n_states >= 2
        u0[1] = total * (1 - frac_r)
        u0[2] = total * frac_r
    end

    return u0
end

"""
    run_sweep(spec::Registry.ModelSpec, params::AbstractVector{<:Real>, grid::SweepGrid;
              exposure_builder::Function = dose -> Exposure.ConstantExposure(Float64(dose)),
              reltol::Float64 = 1e-8, abstol::Float64 = 1e-8, enforce_nonnegative::Bool = true)

Run a parameter sweep simulation across multiple combinations of initial conditions.

This function runs simulations over a grid of initial population sizes, resistant fractions,
and drug doses, collecting summary statistics for each parameter combination.

# Arguments
- `spec::Registry.ModelSpec`: The model specification to simulate
- `params::AbstractVector{<:Real>`: Parameter values for the model (shared across all simulations)
- `grid::SweepGrid`: Defines the parameter sweep space (seed_totals, resistant_fractions, doses, times)
- `exposure_builder::Function = dose -> Exposure.ConstantExposure(Float64(dose))`: 
  Function that converts a dose value to an exposure function
- `reltol::Float64 = 1e-8`: Relative tolerance for the ODE solver
- `abstol::Float64 = 1e-8`: Absolute tolerance for the ODE solver
- `enforce_nonnegative::Bool = true`: Whether to enforce non-negative state variables

# Returns
- `SweepResult`: Contains a summary DataFrame and dictionary of detailed simulation results

# Examples
```julia
# Define a parameter sweep grid
grid = SweepGrid([1.0, 10.0], [0.0, 0.1, 0.5], [0.0, 1.0, 10.0], 0.0:0.5:20.0)

# Run a sweep with a logistic growth model
spec = Registry.get_model("logistic_growth")
params = [0.5, 100.0]  # growth rate, carrying capacity
sweep_result = run_sweep(spec, params, grid)

# Access summary results
summary_df = sweep_result.summary
first_result = sweep_result.simulations[(seed_total=1.0, resistant_fraction=0.0, dose=0.0)]
```
"""
    rows = NamedTuple[]
    sims = Dict{NamedTuple,SimulationResult}()

    for seed_total in grid.seed_totals, resistant_fraction in grid.resistant_fractions, dose in grid.doses
        u0 = _make_sweep_u0(spec, seed_total, resistant_fraction)
        exposure = exposure_builder(dose)
        sim = simulate(
            spec,
            grid.times,
            params;
            u0=u0,
            exposure=exposure,
            reltol=reltol,
            abstol=abstol,
            enforce_nonnegative=enforce_nonnegative,
        )

        key = (seed_total=Float64(seed_total), resistant_fraction=Float64(resistant_fraction), dose=Float64(dose))
        sims[key] = sim

        final_total = sim.success ? sum(sim.states[:, end]) : NaN
        final_resistant_fraction = sim.success && size(sim.states, 1) >= 2 ? sim.states[2, end] / max(sum(sim.states[:, end]), 1e-12) : NaN
        auc_total = sim.success ? sum(sim.observed) : NaN

        push!(rows, (
            seed_total=Float64(seed_total),
            resistant_fraction=Float64(resistant_fraction),
            dose=Float64(dose),
            success=sim.success,
            reason=sim.reason,
            final_total=final_total,
            final_observed=sim.success ? sim.observed[end] : NaN,
            final_resistant_fraction=final_resistant_fraction,
            auc_total=auc_total,
        ))
    end

    return SweepResult(DataFrame(rows), sims)
end

end # module Simulation
