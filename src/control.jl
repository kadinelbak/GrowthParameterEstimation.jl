module Control

using DifferentialEquations
using JSON

using ..Registry

export ControlModel, controlled_model, control_rhs, control_rhs!, mpc_functions,
       discrete_dynamics, InputSchedule, simulate_schedule, schedule_ensemble,
       parameter_sets, control_model_summary, save_control_model

"""
    ControlModel

A fitted model packaged for optimal-control and model-predictive-control tools.

`GrowthParameterEstimation` does not solve control problems itself. A `ControlModel`
is the handoff: it turns a fitted right-hand side into the standard
`f(x, u, p)` form those tools expect, and carries the point estimate and any
bootstrap parameter sets so schedules can be checked for robustness.

The control input `u` replaces the package's exposure function. Inside the
right-hand side, `exposure(t)` returns `u[1]` when the model has one input and
the whole input vector `u` when it has several. Build one with
[`controlled_model`](@ref).

# Fields
- `name::String`: Model name
- `rhs!::Function`: Right-hand side in the package convention `(dx, x, p, t, exposure)`
- `n_states::Int`, `n_inputs::Int`: State and input dimensions
- `state_names`, `input_names`, `param_names`: Labels for states, inputs and parameters
- `params::Vector{Float64}`: Point estimate
- `param_samples::Vector{Vector{Float64}}`: Bootstrap (or other) parameter sets; may be empty
- `observable::Function`: Output map `(x, p) -> y::Vector`
- `n_outputs::Int`, `output_names`: Output dimension and labels
- `autonomous::Bool`: `true` when the right-hand side does not depend on `t` directly
- `metadata::Dict{Symbol,Any}`: Free-form provenance (source model, fit, units)
"""
struct ControlModel
    name::String
    rhs!::Function
    n_states::Int
    n_inputs::Int
    state_names::Vector{Symbol}
    input_names::Vector{Symbol}
    param_names::Vector{Symbol}
    params::Vector{Float64}
    param_samples::Vector{Vector{Float64}}
    observable::Function
    n_outputs::Int
    output_names::Vector{Symbol}
    autonomous::Bool
    metadata::Dict{Symbol,Any}
end

function Base.show(io::IO, cm::ControlModel)
    print(io, "ControlModel(\"", cm.name, "\"; states=", cm.state_names, ", inputs=", cm.input_names,
          ", params=", length(cm.params), ", samples=", length(cm.param_samples), ")")
end

_input_fn(u, n_inputs::Int) = n_inputs == 1 ? (_ -> u[1]) : (_ -> u)

function _call_rhs!(rhs!, dx, x, p, t, exposure)
    if applicable(rhs!, dx, x, p, t, exposure)
        rhs!(dx, x, p, t, exposure)
    else
        rhs!(dx, x, p, t)
    end
    return nothing
end

function _samples(param_samples)
    param_samples === nothing && return Vector{Vector{Float64}}()
    if param_samples isa NamedTuple
        haskey(param_samples, :estimates) ||
            throw(ArgumentError("param_samples NamedTuple must have an `estimates` field (e.g. a bootstrap_joint_fit result)"))
        param_samples = param_samples.estimates
    end
    return [Float64.(collect(s)) for s in param_samples]
end

function _wrap_observable(observable, n_states::Int)
    observable === nothing && return ((x, p) -> collect(x[1:n_states]))
    return (x, p) -> begin
        y = applicable(observable, x, p) ? observable(x, p) : observable(x)
        y isa Number ? [y] : collect(y)
    end
end

"""
    controlled_model(rhs!, params; n_states, n_inputs = 1, kwargs...) -> ControlModel
    controlled_model(spec::ModelSpec, params; kwargs...) -> ControlModel

Package a fitted model for control tools.

The first form takes any right-hand side in the package convention
`rhs!(dx, x, p, t, exposure)` (the same convention as registered models), such as
the model passed to `run_joint_fit`. The second form takes a registered
`ModelSpec` and its fitted parameter vector (for example `fit_model(...).params`).

# Keyword arguments
- `param_samples = nothing`: Bootstrap parameter sets: a vector of vectors, or a
  `bootstrap_joint_fit` result (its `estimates` are used)
- `name`, `state_names`, `input_names`, `param_names`: Labels (defaults are generated)
- `observable = nothing`: Output map `(x, p) -> y` or `x -> y`; default outputs all states
- `output_names = nothing`: Output labels
- `autonomous = true`: Declare whether the right-hand side ignores `t`. Control
  schedules built on non-autonomous models assume the dose started at `t = 0`.
- `metadata = Dict()`: Provenance to carry along (units, source fit, stage)

# Examples
```julia
spec = get_model("logistic_linear_kill")
fit = fit_model(spec, t, y, 1.0)
cm = controlled_model(spec, fit.params; input_names = [:dose])

f = control_rhs(cm)          # f(x, u, p, t) -> dx
F = discrete_dynamics(cm; dt = 2.0) # x_next = F(x, u, p)
```
"""
function controlled_model(
    rhs!::Function,
    params::AbstractVector{<:Real};
    n_states::Integer,
    n_inputs::Integer = 1,
    param_samples = nothing,
    name::AbstractString = "controlled_model",
    state_names::Union{Nothing,AbstractVector{Symbol}} = nothing,
    input_names::Union{Nothing,AbstractVector{Symbol}} = nothing,
    param_names::Union{Nothing,AbstractVector{Symbol}} = nothing,
    observable = nothing,
    output_names::Union{Nothing,AbstractVector{Symbol}} = nothing,
    autonomous::Bool = true,
    metadata::AbstractDict = Dict{Symbol,Any}(),
)
    nx = Int(n_states)
    nu = Int(n_inputs)
    nx >= 1 || throw(ArgumentError("n_states must be >= 1"))
    nu >= 1 || throw(ArgumentError("n_inputs must be >= 1"))
    p = Float64.(collect(params))

    s_names = state_names === nothing ? [Symbol("x", i) for i in 1:nx] : collect(state_names)
    i_names = input_names === nothing ? (nu == 1 ? [:u] : [Symbol("u", i) for i in 1:nu]) : collect(input_names)
    p_names = param_names === nothing ? [Symbol("p", i) for i in eachindex(p)] : collect(param_names)
    length(s_names) == nx || throw(ArgumentError("state_names must have length n_states = $nx"))
    length(i_names) == nu || throw(ArgumentError("input_names must have length n_inputs = $nu"))
    length(p_names) == length(p) || throw(ArgumentError("param_names must have length $(length(p))"))

    samples = _samples(param_samples)
    for s in samples
        length(s) == length(p) || throw(ArgumentError("every parameter sample must have length $(length(p))"))
    end

    obs = _wrap_observable(observable, nx)
    x_probe = ones(nx)
    u_probe = zeros(nu)
    y_probe = try
        obs(x_probe, p)
    catch err
        throw(ArgumentError("observable failed on a test state: $(sprint(showerror, err))"))
    end
    ny = length(y_probe)
    o_names = output_names === nothing ? (observable === nothing ? copy(s_names) : [Symbol("y", i) for i in 1:ny]) : collect(output_names)
    length(o_names) == ny || throw(ArgumentError("output_names must have length $ny"))

    dx_probe = zeros(nx)
    try
        _call_rhs!(rhs!, dx_probe, x_probe, p, 0.0, _input_fn(u_probe, nu))
    catch err
        throw(ArgumentError("rhs! failed on a test call with $nx states and $nu inputs: $(sprint(showerror, err))"))
    end

    meta = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in metadata)
    return ControlModel(String(name), rhs!, nx, nu, s_names, i_names, p_names, p, samples,
                        obs, ny, o_names, autonomous, meta)
end

function controlled_model(
    spec::Registry.ModelSpec,
    params::AbstractVector{<:Real};
    n_inputs::Integer = 1,
    name::AbstractString = spec.name,
    state_names = spec.state_names,
    param_names = spec.param_names,
    observable = nothing,
    output_names = nothing,
    metadata::AbstractDict = Dict{Symbol,Any}(),
    kwargs...,
)
    length(params) == length(spec.param_names) ||
        throw(ArgumentError("params must have length $(length(spec.param_names)) for model $(spec.name)"))
    obs = observable === nothing ? (x -> spec.observable(x)) : observable
    o_names = output_names === nothing && observable === nothing ? [:observed] : output_names
    meta = Dict{Symbol,Any}(:source_model => spec.name, :family => spec.base_growth_family)
    for (k, v) in metadata
        meta[Symbol(k)] = v
    end
    return controlled_model(spec.ode!, params; n_states = spec.n_states, n_inputs = n_inputs, name = name,
                            state_names = state_names, param_names = param_names, observable = obs,
                            output_names = o_names, metadata = meta, kwargs...)
end

"""
    control_rhs!(cm::ControlModel) -> (dx, x, u, p, t) -> nothing

In-place continuous-time dynamics `dx = f(x, u, p, t)` with the control input `u`
held fixed. Compatible with ForwardDiff dual numbers in `x`, `u` and `p`.
"""
function control_rhs!(cm::ControlModel)
    rhs! = cm.rhs!
    nu = cm.n_inputs
    return function (dx, x, u, p, t)
        fill!(dx, zero(eltype(dx)))
        _call_rhs!(rhs!, dx, x, p, t, _input_fn(u, nu))
        return nothing
    end
end

"""
    control_rhs(cm::ControlModel) -> (x, u, p = cm.params, t = 0.0) -> dx

Out-of-place continuous-time dynamics. The returned vector takes the promoted
element type of `x`, `u` and `p`, so it can be differentiated with ForwardDiff.
"""
function control_rhs(cm::ControlModel)
    f! = control_rhs!(cm)
    default_p = cm.params
    return function (x, u, p = default_p, t = 0.0)
        T = promote_type(eltype(x), eltype(u), eltype(p))
        dx = zeros(T, length(x))
        f!(dx, x, u, p, t)
        return dx
    end
end

"""
    mpc_functions(cm::ControlModel) -> NamedTuple

Dynamics and output functions in the `(ẋ, x, u, d, p)` / `(y, x, d, p)`
convention used by nonlinear MPC packages such as ModelPredictiveControl.jl
(`NonLinModel(f!, h!, Ts, nu, nx, ny; p = p)`). `d` (measured disturbances) is
accepted and ignored.

Returns `(f!, h!, nx, nu, ny, p)`, where `p` is the point estimate.
"""
function mpc_functions(cm::ControlModel)
    rhs! = control_rhs!(cm)
    obs = cm.observable
    f! = (xdot, x, u, d, p) -> (rhs!(xdot, x, u, p, 0.0); nothing)
    h! = (y, x, d, p) -> (y .= obs(x, p); nothing)
    return (f! = f!, h! = h!, nx = cm.n_states, nu = cm.n_inputs, ny = cm.n_outputs, p = copy(cm.params))
end

"""
    discrete_dynamics(cm::ControlModel; dt, substeps = 20) -> (x, u, p = cm.params, t0 = 0.0) -> x_next

Discrete-time transition map for a zero-order hold: the state after holding the
input `u` constant for `dt`, integrated with `substeps` fixed classical RK4 steps.
This is the step map used by direct shooting and MPC solvers. It allocates no
solver state and differentiates cleanly with ForwardDiff, so Jacobians
`∂F/∂x` and `∂F/∂u` can be taken directly.
"""
function discrete_dynamics(cm::ControlModel; dt::Real, substeps::Integer = 20)
    dt > 0 || throw(ArgumentError("dt must be positive"))
    substeps >= 1 || throw(ArgumentError("substeps must be >= 1"))
    f = control_rhs(cm)
    h = Float64(dt) / substeps
    n = Int(substeps)
    default_p = cm.params
    return function (x, u, p = default_p, t0 = 0.0)
        T = promote_type(eltype(x), eltype(u), eltype(p))
        xk = T.(collect(x))
        t = Float64(t0)
        for _ in 1:n
            k1 = f(xk, u, p, t)
            k2 = f(xk .+ (h / 2) .* k1, u, p, t + h / 2)
            k3 = f(xk .+ (h / 2) .* k2, u, p, t + h / 2)
            k4 = f(xk .+ h .* k3, u, p, t + h)
            xk = xk .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
            t += h
        end
        return xk
    end
end

"""
    InputSchedule(change_times, values)

A piecewise-constant input schedule. `values[i]` (a scalar, or a vector with one
entry per input) is applied from `change_times[i]` until `change_times[i+1]`; the
last value is held to the end of the simulation. This is the form an optimal
control solution takes when doses change only at decision times.

# Examples
```julia
# 1 µM on days 6–12 of a 14-day horizon, decisions every 2 days
sched = InputSchedule(0:2:12, [0, 0, 0, 1.0, 1.0, 1.0, 0])
```
"""
struct InputSchedule
    change_times::Vector{Float64}
    values::Vector{Vector{Float64}}
    function InputSchedule(change_times, values)
        ts = Float64.(collect(change_times))
        vs = [v isa Number ? [Float64(v)] : Float64.(collect(v)) for v in values]
        length(ts) == length(vs) || throw(ArgumentError("change_times and values must have the same length"))
        isempty(ts) && throw(ArgumentError("schedule cannot be empty"))
        issorted(ts) || throw(ArgumentError("change_times must be sorted"))
        allequal(length.(vs)) || throw(ArgumentError("every schedule value must have the same length"))
        return new(ts, vs)
    end
end

function (s::InputSchedule)(t::Real)
    i = searchsortedlast(s.change_times, t)
    return s.values[max(i, 1)]
end

"""
    simulate_schedule(cm, x0, schedule, times; params = cm.params, solver = Tsit5(),
                      reltol = 1e-8, abstol = 1e-8) -> NamedTuple

Simulate the model under a piecewise-constant [`InputSchedule`](@ref) and return
`(success, times, states, outputs, inputs)`. `states` is `n_states × length(times)`,
`outputs` is `n_outputs × length(times)`. The solver stops exactly at every input
change, so jumps in the dose are resolved. Use it to replay a schedule chosen by
an external optimizer, or comparator schedules such as constant dosing.
"""
function simulate_schedule(
    cm::ControlModel,
    x0::AbstractVector{<:Real},
    schedule::InputSchedule,
    times::AbstractVector{<:Real};
    params::AbstractVector{<:Real} = cm.params,
    solver = Tsit5(),
    reltol::Real = 1e-8,
    abstol::Real = 1e-8,
)
    length(x0) == cm.n_states || throw(ArgumentError("x0 must have length $(cm.n_states)"))
    length(first(schedule.values)) == cm.n_inputs ||
        throw(ArgumentError("schedule values must have length n_inputs = $(cm.n_inputs)"))
    ts = Float64.(collect(times))
    issorted(ts) || throw(ArgumentError("times must be sorted"))
    f! = control_rhs!(cm)
    ode! = (dx, x, p, t) -> f!(dx, x, schedule(t), p, t)
    tspan = (ts[1], ts[end])
    stops = filter(t -> tspan[1] < t < tspan[2], schedule.change_times)
    prob = ODEProblem(ode!, Float64.(collect(x0)), tspan, Float64.(collect(params)))
    sol = solve(prob, solver; saveat = ts, tstops = stops, reltol = reltol, abstol = abstol)
    inputs = stack([schedule(t) for t in ts])
    if sol.retcode != ReturnCode.Success || length(sol.u) != length(ts)
        nanm(n) = fill(NaN, n, length(ts))
        return (success = false, times = ts, states = nanm(cm.n_states), outputs = nanm(cm.n_outputs), inputs = inputs)
    end
    states = stack(sol.u)
    outputs = stack([cm.observable(sol.u[i], params) for i in eachindex(ts)])
    return (success = true, times = ts, states = states, outputs = outputs, inputs = inputs)
end

"""
    parameter_sets(cm::ControlModel; include_estimate = true) -> Vector{Vector{Float64}}

The point estimate (first, when `include_estimate`) followed by every parameter sample.
"""
function parameter_sets(cm::ControlModel; include_estimate::Bool = true)
    sets = [copy(s) for s in cm.param_samples]
    include_estimate && pushfirst!(sets, copy(cm.params))
    return sets
end

"""
    schedule_ensemble(cm, x0, schedule, times; include_estimate = true, kwargs...) -> Vector

Replay one schedule on every parameter set from [`parameter_sets`](@ref) and return
one [`simulate_schedule`](@ref) result per set. This is the robustness check for
a schedule chosen with the point estimate: how far outcomes spread when the
parameters are drawn from their bootstrap distribution.
"""
function schedule_ensemble(cm::ControlModel, x0, schedule::InputSchedule, times; include_estimate::Bool = true, kwargs...)
    return [simulate_schedule(cm, x0, schedule, times; params = p, kwargs...)
            for p in parameter_sets(cm; include_estimate = include_estimate)]
end

"""
    control_model_summary(cm::ControlModel) -> Dict{String,Any}

Plain-data description of the model (names, dimensions, estimates, samples,
metadata) for handing to other tools or languages. Functions are not included.
"""
function control_model_summary(cm::ControlModel)
    return Dict{String,Any}(
        "name" => cm.name,
        "n_states" => cm.n_states,
        "n_inputs" => cm.n_inputs,
        "n_outputs" => cm.n_outputs,
        "state_names" => String.(cm.state_names),
        "input_names" => String.(cm.input_names),
        "output_names" => String.(cm.output_names),
        "param_names" => String.(cm.param_names),
        "params" => cm.params,
        "param_samples" => cm.param_samples,
        "autonomous" => cm.autonomous,
        "metadata" => Dict{String,Any}(String(k) => (v isa Union{Number,AbstractString,Bool,AbstractVector} ? v : string(v))
                                       for (k, v) in cm.metadata),
    )
end

"""
    save_control_model(cm::ControlModel, path) -> path

Write [`control_model_summary`](@ref) to `path` as JSON.
"""
function save_control_model(cm::ControlModel, path::AbstractString)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        JSON.print(io, control_model_summary(cm), 2)
    end
    return path
end

end # module Control
