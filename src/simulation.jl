module Simulation

using DifferentialEquations

using ..Exposure
using ..Registry

export SimulationResult, simulate, classify_failure

struct SimulationResult
    success::Bool
    reason::String
    times::Vector{Float64}
    states::Matrix{Float64}
    observed::Vector{Float64}
end

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

function _solver_from_symbol(kind::Symbol)
    if kind == :ode
        return Tsit5()
    elseif kind == :stiff_ode
        return Rodas5()
    else
        return Tsit5()
    end
end

function _to_state_matrix(sol)
    n_t = length(sol.u)
    n_s = length(sol.u[1])
    arr = zeros(n_s, n_t)
    for i in 1:n_t
        for j in 1:n_s
            arr[j, i] = sol.u[i][j]
        end
    end
    return arr
end

function simulate(
    spec::Registry.ModelSpec,
    times::AbstractVector{<:Real},
    params::AbstractVector{<:Real};
    u0::AbstractVector{<:Real},
    exposure::Exposure.AbstractExposure = Exposure.ConstantExposure(0.0),
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-8,
    enforce_nonnegative::Bool = true,
)
    t_obs = Float64.(collect(times))
    tspan = (minimum(t_obs), maximum(t_obs))

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

    prob = ODEProblem(wrapped!, Float64.(u0), tspan, Float64.(params))

    try
        solver = _solver_from_symbol(spec.solver_type)
        sol = solve(prob, solver; saveat=t_obs, reltol=reltol, abstol=abstol, isoutofdomain=(u, p, t) -> any(x -> x < -1e-10 || !isfinite(x), u))

        if sol.retcode != ReturnCode.Success
            return SimulationResult(false, "solver_failure: $(sol.retcode)", t_obs, zeros(length(u0), length(t_obs)), fill(NaN, length(t_obs)))
        end

        state_matrix = _to_state_matrix(sol)
        if enforce_nonnegative
            state_matrix .= max.(state_matrix, 0.0)
        end

        observed = [spec.observation(state_matrix[:, i], params, t_obs[i]) for i in eachindex(t_obs)]
        return SimulationResult(true, "ok", t_obs, state_matrix, observed)
    catch err
        return SimulationResult(false, classify_failure(err), t_obs, zeros(length(u0), length(t_obs)), fill(NaN, length(t_obs)))
    end
end

end # module Simulation
