module Registry

using DifferentialEquations

using ..Models

export ModelSpec, register_model!, register_model, register_models_from_file!, get_model, list_models, models_by_family, clear_registry!, register_builtin_models!, register_composable_model

"""
    ModelSpec

Public model specification used for registration, simulation, and fitting.

Primary fields:
- `name`, `ode!`, `param_names`, `bounds`, `n_states`, `observable`
- `base_growth_family`, `default_solver`, `p0_factory`, `fixed_params`

Compatibility fields are kept so legacy workflow/simulation APIs continue to work:
- `dynamics!`, `state_names`, `observation`, `solver_type`, `metadata`
"""
struct ModelSpec
    name::String
    ode!::Function
    param_names::Vector{Symbol}
    bounds::Vector{Tuple{Float64,Float64}}
    n_states::Int
    observable::Function
    base_growth_family::String
    default_solver
    p0_factory::Union{Function,Nothing}
    fixed_params::Dict{Int,Float64}

    # Legacy compatibility fields
    dynamics!::Function
    state_names::Vector{Symbol}
    observation::Function
    solver_type::Symbol
    metadata::Dict{Symbol,Any}
end

const MODEL_REGISTRY = Dict{String,ModelSpec}()

function _solver_to_symbol(solver)
    if solver isa Rodas5 || solver isa Rosenbrock23 || solver isa TRBDF2
        return :stiff_ode
    end
    return :ode
end

function _solver_from_symbol(kind::Symbol)
    if kind == :stiff_ode
        return Rodas5()
    end
    return Tsit5()
end

function _normalize_bounds(bounds)
    return [(Float64(first(b)), Float64(last(b))) for b in bounds]
end

function _default_state_names(n_states::Int)
    return [Symbol("x", i) for i in 1:n_states]
end

"""
    ModelSpec(; ...)

Preferred constructor for custom model registration.
"""
function ModelSpec(
    ;
    name::AbstractString,
    ode!::Function,
    param_names::AbstractVector{Symbol},
    bounds::AbstractVector,
    n_states::Integer,
    observable::Function = u -> u[1],
    base_growth_family::AbstractString = "custom",
    default_solver = Tsit5(),
    p0_factory::Union{Function,Nothing} = nothing,
    fixed_params::Dict{Int,<:Real} = Dict{Int,Float64}(),
    state_names::Union{Nothing,AbstractVector{Symbol}} = nothing,
    metadata::AbstractDict{Symbol,<:Any} = Dict{Symbol,Any}(),
)
    pname_vec = collect(param_names)
    bounds_vec = _normalize_bounds(bounds)
    length(pname_vec) == length(bounds_vec) || throw(ArgumentError("param_names and bounds length mismatch"))
    n_states_int = Int(n_states)
    n_states_int >= 1 || throw(ArgumentError("n_states must be >= 1"))

    s_names = isnothing(state_names) ? _default_state_names(n_states_int) : collect(state_names)
    length(s_names) == n_states_int || throw(ArgumentError("state_names length must equal n_states"))

    obs = (u, p, t) -> observable(u)
    solver_sym = _solver_to_symbol(default_solver)
    meta = Dict{Symbol,Any}(k => v for (k, v) in metadata)
    meta[:family] = String(base_growth_family)
    fixed_cast = Dict{Int,Float64}(k => Float64(v) for (k, v) in fixed_params)

    return ModelSpec(
        String(name),
        ode!,
        pname_vec,
        bounds_vec,
        n_states_int,
        observable,
        String(base_growth_family),
        default_solver,
        p0_factory,
        fixed_cast,
        ode!,
        s_names,
        obs,
        solver_sym,
        meta,
    )
end

"""
    ModelSpec(name, dynamics!, param_names, bounds, state_names, observation, solver_type, metadata)

Legacy positional constructor maintained for backward compatibility.
"""
function ModelSpec(
    name::String,
    dynamics!::Function,
    param_names::Vector{Symbol},
    bounds::Vector{Tuple{Float64,Float64}},
    state_names::Vector{Symbol},
    observation::Function,
    solver_type::Symbol,
    metadata::AbstractDict{Symbol,<:Any},
)
    base_family = haskey(metadata, :family) ? String(metadata[:family]) : "legacy"
    solver = _solver_from_symbol(solver_type)
    observable = u -> observation(u, nothing, 0.0)

    spec = ModelSpec(
        ;
        name=name,
        ode! = dynamics!,
        param_names=param_names,
        bounds=bounds,
        n_states=length(state_names),
        observable=observable,
        base_growth_family=base_family,
        default_solver=solver,
        p0_factory=nothing,
        fixed_params=Dict{Int,Float64}(),
        state_names=state_names,
        metadata=Dict{Symbol,Any}(k => v for (k, v) in metadata),
    )

    return ModelSpec(
        spec.name,
        spec.ode!,
        spec.param_names,
        spec.bounds,
        spec.n_states,
        spec.observable,
        spec.base_growth_family,
        spec.default_solver,
        spec.p0_factory,
        spec.fixed_params,
        dynamics!,
        state_names,
        observation,
        solver_type,
        Dict{Symbol,Any}(k => v for (k, v) in metadata),
    )
end

"""
    register_model!(spec::ModelSpec; overwrite=false) -> nothing

Register a model specification in the central registry.
"""
function register_model!(spec::ModelSpec; overwrite::Bool=false)
    if haskey(MODEL_REGISTRY, spec.name) && !overwrite
        error("Model $(spec.name) already registered. Use overwrite=true to replace it.")
    end
    MODEL_REGISTRY[spec.name] = spec
    return nothing
end

"""
    register_model(spec::ModelSpec; overwrite=false) -> ModelSpec

Compatibility wrapper returning the registered spec.
"""
function register_model(spec::ModelSpec; overwrite::Bool=false)
    register_model!(spec; overwrite=overwrite)
    return spec
end

"""
    register_models_from_file!(path; target_module=Main, register_fn=:register_custom_models!, overwrite=false)

Load a user model file and register custom models.

The file can either:
1) call `register_model!` directly at top-level, or
2) define `register_custom_models!()` (or another function via `register_fn`) that performs registration.
"""
function register_models_from_file!(
    path::AbstractString;
    target_module::Module = Main,
    register_fn::Symbol = :register_custom_models!,
    overwrite::Bool = false,
)
    isfile(path) || throw(ArgumentError("Model file not found: $(path)"))

    Base.include(target_module, path)

    if Base.invokelatest(isdefined, target_module, register_fn)
        fn = Base.invokelatest(getfield, target_module, register_fn)
        if fn isa Function
            try
                Base.invokelatest(fn; overwrite=overwrite)
            catch err
                if err isa MethodError
                    Base.invokelatest(fn)
                else
                    rethrow(err)
                end
            end
        end
    end

    return list_models()
end

"""
    get_model(name::String) -> ModelSpec
"""
get_model(name::String) = MODEL_REGISTRY[name]

"""
    list_models() -> Vector{String}
"""
list_models() = sort(collect(keys(MODEL_REGISTRY)))

"""
    models_by_family(family::String) -> Vector{ModelSpec}
"""
function models_by_family(family::String)
    fam = lowercase(strip(family))
    specs = [spec for spec in values(MODEL_REGISTRY) if lowercase(spec.base_growth_family) == fam]
    sort!(specs, by = s -> s.name)
    return specs
end

clear_registry!() = empty!(MODEL_REGISTRY)



function _logistic_linear_kill!(du, u, p, t, exposure)
    r, K, kill_coeff = p
    N = max(u[1], 0.0)
    dose = max(exposure(t), 0.0)
    du[1] = r * N * (1 - N / max(K, 1e-8)) - kill_coeff * dose * N
end

function _theta_hill_inhibition!(du, u, p, t, exposure)
    r, K, theta, ic50, hill = p
    N = max(u[1], 0.0)
    drug = max(exposure(t), 0.0)
    inhibition = drug^hill / (ic50^hill + drug^hill + 1e-12)
    growth = r * N * (1 - (N / max(K, 1e-8))^theta)
    du[1] = max(1e-12, 1 - inhibition) * growth
end

function _theta_hill_kill!(du, u, p, t, exposure)
    r, K, theta, emax_kill, ic50, hill = p
    N = max(u[1], 0.0)
    drug = max(exposure(t), 0.0)
    kill = emax_kill * (drug^hill / (ic50^hill + drug^hill + 1e-12))
    growth = r * N * (1 - (N / max(K, 1e-8))^theta)
    du[1] = growth - kill * N
end

function _sensitive_resistant!(du, u, p, t, exposure)
    rS, rR, K, kSR, emax, ic50, hill = p
    S = max(u[1], 0.0)
    R = max(u[2], 0.0)
    N = max(S + R, 1e-12)
    drug = max(exposure(t), 0.0)
    killS = emax * (drug^hill / (ic50^hill + drug^hill + 1e-12))
    common = max(0.0, 1 - N / max(K, 1e-8))
    du[1] = rS * S * common - killS * S - kSR * S
    du[2] = rR * R * common + kSR * S
end

function _null_coculture!(du, u, p, t, exposure)
    rS, KS, rR, KR = p
    S = max(u[1], 0.0)
    R = max(u[2], 0.0)
    du[1] = rS * S * max(0.0, 1 - S / max(KS, 1e-8))
    du[2] = rR * R * max(0.0, 1 - R / max(KR, 1e-8))
end

function _lotka_volterra_competition!(du, u, p, t, exposure)
    rS, KS, alpha_SR, rR, KR, alpha_RS = p
    S = max(u[1], 0.0)
    R = max(u[2], 0.0)
    du[1] = rS * S * max(0.0, 1 - (S + alpha_SR * R) / max(KS, 1e-8))
    du[2] = rR * R * max(0.0, 1 - (R + alpha_RS * S) / max(KR, 1e-8))
end

function _lotka_volterra_hill_competition!(du, u, p, t, exposure)
    rS, KS, alpha_SR, rR, KR, alpha_RS, emaxS, ic50S, emaxR, ic50R, hill = p
    S = max(u[1], 0.0)
    R = max(u[2], 0.0)
    drug = max(exposure(t), 0.0)
    effectS = emaxS * (drug^hill / (ic50S^hill + drug^hill + 1e-12))
    effectR = emaxR * (drug^hill / (ic50R^hill + drug^hill + 1e-12))
    growthS = rS * S * max(0.0, 1 - (S + alpha_SR * R) / max(KS, 1e-8))
    growthR = rR * R * max(0.0, 1 - (R + alpha_RS * S) / max(KR, 1e-8))
    du[1] = growthS - effectS * S
    du[2] = growthR - effectR * R
end

function _damage_repair_arrest!(du, u, p, t, exposure)
    r, K, k_damage, k_repair, k_death, ic50, hill = p
    S = max(u[1], 0.0)
    D = max(u[2], 0.0)
    N = max(S + D, 1e-12)
    drug = max(exposure(t), 0.0)
    effect = drug^hill / (ic50^hill + drug^hill + 1e-12)
    growth = r * S * max(0.0, 1 - N / max(K, 1e-8))
    damage_flux = k_damage * effect * S
    repair_flux = k_repair * D
    du[1] = growth - damage_flux + repair_flux
    du[2] = damage_flux - repair_flux - k_death * D
end

function _adaptive_ic50!(du, u, p, t, exposure)
    r, K, emax, ic50_0, hill, k_adapt = p
    N = max(u[1], 0.0)
    A = max(u[2], 0.0)
    ic50_t = ic50_0 * (1 + A)
    drug = max(exposure(t), 0.0)
    inhibition = emax * (drug^hill / (ic50_t^hill + drug^hill + 1e-12))
    growth = r * N * max(0.0, 1 - N / max(K, 1e-8))
    du[1] = growth * max(1e-12, 1 - inhibition)
    du[2] = k_adapt * inhibition - 0.05 * A
end

function _pkpd_inhibition!(du, u, p, t, exposure)
    r, K, emax, ic50, hill, k_elim, k_in = p
    N = max(u[1], 0.0)
    C = max(u[2], 0.0)
    inhibition = emax * (C^hill / (ic50^hill + C^hill + 1e-12))
    du[1] = r * N * max(0.0, 1 - N / max(K, 1e-8)) * max(1e-12, 1 - inhibition)
    du[2] = k_in * max(exposure(t), 0.0) - k_elim * C
end

function _bi_exponential!(du, u, p, t, exposure)
    a, b = p
    du[1] = -a * u[1]
    du[2] = -b * u[2]
end

function _transit_chain_erlang!(du, u, p, t, exposure)
    r, K, ktr, emax, ic50, hill = p
    N = max(u[1], 0.0)
    X1 = max(u[2], 0.0)
    X2 = max(u[3], 0.0)
    X3 = max(u[4], 0.0)
    drug = max(exposure(t), 0.0)
    effect = emax * (drug^hill / (ic50^hill + drug^hill + 1e-12))
    growth = r * N * max(0.0, 1 - N / max(K, 1e-8))
    du[1] = growth - ktr * X3
    du[2] = effect * N - ktr * X1
    du[3] = ktr * (X1 - X2)
    du[4] = ktr * (X2 - X3)
end

# ---------------------------------------------------------------------------
# Helper functions for automatic ModelSpec generation from composable models
# ---------------------------------------------------------------------------
"""
    _extract_param_names(model::AbstractBaseModel)

Extract parameter names from a composable model's fields.
"""
function _extract_param_names(model::AbstractBaseModel)
    return fieldnames(typeof(model))
end

"""
    _detect_n_states(model::AbstractBaseModel)

Attempt to detect the number of states from a model.
For most base models, this is 1 state.
"""
function _detect_n_states(model::AbstractBaseModel)
    # Default assumption for simple growth models
    return 1
end

"""
    _suggest_default_bounds(model::AbstractBaseModel)

Suggest reasonable default bounds for model parameters.
"""
function _suggest_default_bounds(model::AbstractBaseModel)
    param_names = _extract_param_names(model)
    bounds = Tuple{Float64,Float64}[]
    
    for param in param_names
        param_str = string(param)
        if param_str in ("r", "a")  # growth rates
            push!(bounds, (1e-6, 5.0))
        elseif param_str in ("K",)   # carrying capacity
            push!(bounds, (1e-3, 1e7))
        elseif param_str in ("death_rate", "emax", "kill_coeff", "emax_kill")  # death/inhibition rates
            push!(bounds, (0.0, 20.0))
        elseif param_str in ("tlag", "hill")  # time delay / Hill coefficient
            push!(bounds, (0.0, 10.0))
        elseif param_str in ("ic50", "KR", "KS", "KR", "IC50")  # concentrations / EC50
            push!(bounds, (1e-8, 1e4))
        elseif param_str in ("ktr", "k_damage", "k_repair", "k_death", "k_adapt", "k_elim", "k_in")  # rate constants
            push!(bounds, (1e-6, 10.0))
        elseif param_str in ("theta", "b")  # shape parameters
            push!(bounds, (0.1, 5.0))
        else
            push!(bounds, (1e-6, 10.0))  # sensible default
        end
    end
    
    return bounds
end

"""
    composable_model_spec(; name, model, bounds=nothing, observable=u->u[1], 
                          default_solver=Tsit5(), kwargs...)

Create a ModelSpec from a composable model, automatically extracting
parameter names and suggesting bounds.
"""
function composable_model_spec(; 
                               name::AbstractString,
                               model::AbstractBaseModel,
                               bounds=nothing,
                               observable::Function = u -> u[1],
                               default_solver = Tsit5(),
                               kwargs...)
    # Auto-extract information from the model
    param_names = _extract_param_names(model)
    n_states = _detect_n_states(model)
    
    # Use provided bounds or suggest defaults
    if isnothing(bounds)
        bounds = _suggest_default_bounds(model)
    end
    
    # Convert the model to an ODE function
    ode_func = Models.to_ode!(model)
    
    # Process metadata from kwargs
    meta = Dict{Symbol,Any}(kwargs)
    base_family = get(meta, :family, "custom")
    delete!(meta, :family)  # Remove family from metadata as it's stored separately
    
    # Create the ModelSpec
    return ModelSpec(
        String(name),
        ode_func,
        collect(param_names),
        [(Float64(first(b)), Float64(last(b))) for b in bounds],
        Int(n_states),
        observable,
        String(base_family),
        default_solver,
        nothing,  # p0_factory
        Dict{Int,Float64}(),  # fixed_params
        _default_state_names(n_states),  # state_names
        Models.to_ode!(model),  # dynamics! (legacy)
        _default_state_names(n_states),  # state_names (legacy)
        observable,  # observation (legacy)
        _solver_to_symbol(default_solver),  # solver_type (legacy)
        meta  # metadata
    )
end

"""
    register_composable_model(name::String, model::AbstractBaseModel;
                              bounds=nothing, observable=u->u[1],
                              default_solver=Tsit5(), kwargs...)

Convenience function to register a composable model.
Automatically extracts parameter information and creates a ModelSpec.
"""
function register_composable_model(name::String, model::AbstractBaseModel;
                                  bounds=nothing,
                                  observable::Function = u -> u[1],
                                  default_solver = Tsit5(),
                                  kwargs...)
    spec = composable_model_spec(; name=name, model=model, bounds=bounds, 
                                 observable=observable, default_solver=default_solver, kwargs...)
    return register_model!(spec)
end

function register_builtin_models!()
    clear_registry!()

    register_model!(ModelSpec(
        name="logistic_linear_kill",
        ode! = _logistic_linear_kill!,
        param_names=[:r, :K, :kill_coeff],
        bounds=[(1e-6, 5.0), (1e-3, 1e7), (0.0, 5.0)],
        n_states=1,
        observable=u -> u[1],
        base_growth_family="logistic",
        default_solver=Tsit5(),
        state_names=[:N],
        metadata=Dict(:family => :baseline_kill),
    ))

    register_model!(composable_model_spec(
        name="logistic_growth",
        model=build_logistic(),
        bounds=[(1e-6, 5.0), (1e-3, 1e7)],
        observable=u -> u[1],
        base_growth_family="logistic",
        default_solver=Tsit5(),
        metadata=Dict(:family => :baseline),
    ))

    register_model!(composable_model_spec(
        name="gompertz_growth",
        model=build_gompertz(),
        bounds=[(1e-6, 5.0), (1e-6, 10.0), (1e-3, 1e7)],
        observable=u -> u[1],
        base_growth_family="gompertz",
        default_solver=Tsit5(),
        metadata=Dict(:family => :baseline),
    ))

    register_model!(ModelSpec(
        name="theta_logistic_hill_inhibition",
        ode! = _theta_hill_inhibition!,
        param_names=[:r, :K, :theta, :ic50, :hill],
        bounds=[(1e-6, 5.0), (1e-3, 1e7), (0.1, 5.0), (1e-8, 1e4), (0.1, 8.0)],
        n_states=1,
        observable=u -> u[1],
        base_growth_family="theta_logistic",
        default_solver=Tsit5(),
        state_names=[:N],
        metadata=Dict(:family => :baseline_hill),
    ))

    register_model!(ModelSpec(
        name="theta_logistic_hill_kill",
        ode! = _theta_hill_kill!,
        param_names=[:r, :K, :theta, :emax_kill, :ic50, :hill],
        bounds=[(1e-6, 5.0), (1e-3, 1e7), (0.1, 5.0), (0.0, 20.0), (1e-8, 1e4), (0.1, 8.0)],
        n_states=1,
        observable=u -> u[1],
        base_growth_family="theta_logistic",
        default_solver=Tsit5(),
        state_names=[:N],
        metadata=Dict(:family => :baseline_hill),
    ))

    register_model!(ModelSpec(
        name="null_coculture",
        ode! = _null_coculture!,
        param_names=[:rS, :KS, :rR, :KR],
        bounds=[(1e-6, 5.0), (1e-3, 1e7), (1e-6, 5.0), (1e-3, 1e7)],
        n_states=2,
        observable=u -> u[1] + u[2],
        base_growth_family="coculture",
        default_solver=Tsit5(),
        state_names=[:S, :R],
        metadata=Dict(:family => :coculture_null),
    ))

    register_model!(ModelSpec(
        name="lotka_volterra_competition",
        ode! = _lotka_volterra_competition!,
        param_names=[:rS, :KS, :alpha_SR, :rR, :KR, :alpha_RS],
        bounds=[(1e-6, 5.0), (1e-3, 1e7), (0.0, 5.0), (1e-6, 5.0), (1e-3, 1e7), (0.0, 5.0)],
        n_states=2,
        observable=u -> u[1] + u[2],
        base_growth_family="coculture",
        default_solver=Tsit5(),
        state_names=[:S, :R],
        metadata=Dict(:family => :coculture_competition),
    ))

    register_model!(ModelSpec(
        name="lotka_volterra_hill_competition",
        ode! = _lotka_volterra_hill_competition!,
        param_names=[:rS, :KS, :alpha_SR, :rR, :KR, :alpha_RS, :emaxS, :ic50S, :emaxR, :ic50R, :hill],
        bounds=[(1e-6, 5.0), (1e-3, 1e7), (0.0, 5.0), (1e-6, 5.0), (1e-3, 1e7), (0.0, 5.0), (0.0, 20.0), (1e-8, 1e4), (0.0, 20.0), (1e-8, 1e4), (0.1, 8.0)],
        n_states=2,
        observable=u -> u[1] + u[2],
        base_growth_family="coculture",
        default_solver=Tsit5(),
        state_names=[:S, :R],
        metadata=Dict(:family => :coculture_competition_hill),
    ))

    register_model!(ModelSpec(
        name="sensitive_resistant",
        ode! = _sensitive_resistant!,
        param_names=[:rS, :rR, :K, :kSR, :emax, :ic50, :hill],
        bounds=[(1e-6, 5.0), (1e-6, 5.0), (1e-3, 1e7), (0.0, 2.0), (0.0, 20.0), (1e-8, 1e4), (0.1, 8.0)],
        n_states=2,
        observable=u -> u[1] + u[2],
        base_growth_family="mechanistic",
        default_solver=Tsit5(),
        state_names=[:S, :R],
        metadata=Dict(:family => :mechanistic),
    ))

    register_model!(ModelSpec(
        name="damage_repair_arrest",
        ode! = _damage_repair_arrest!,
        param_names=[:r, :K, :k_damage, :k_repair, :k_death, :ic50, :hill],
        bounds=[(1e-6, 5.0), (1e-3, 1e7), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (1e-8, 1e4), (0.1, 8.0)],
        n_states=2,
        observable=u -> u[1] + u[2],
        base_growth_family="mechanistic",
        default_solver=Tsit5(),
        state_names=[:S, :D],
        metadata=Dict(:family => :mechanistic),
    ))

    register_model!(ModelSpec(
        name="adaptive_ic50",
        ode! = _adaptive_ic50!,
        param_names=[:r, :K, :emax, :ic50_0, :hill, :k_adapt],
        bounds=[(1e-6, 5.0), (1e-3, 1e7), (0.0, 10.0), (1e-8, 1e4), (0.1, 8.0), (0.0, 5.0)],
        n_states=2,
        observable=u -> u[1],
        base_growth_family="adaptive",
        default_solver=Tsit5(),
        state_names=[:N, :A],
        metadata=Dict(:family => :adaptive),
    ))

    register_model!(ModelSpec(
        name="pkpd_inhibition",
        ode! = _pkpd_inhibition!,
        param_names=[:r, :K, :emax, :ic50, :hill, :k_elim, :k_in],
        bounds=[(1e-6, 5.0), (1e-3, 1e7), (0.0, 10.0), (1e-8, 1e4), (0.1, 8.0), (1e-6, 10.0), (0.0, 20.0)],
        n_states=2,
        observable=u -> u[1],
        base_growth_family="pkpd",
        default_solver=Tsit5(),
        state_names=[:N, :C],
        metadata=Dict(:family => :pkpd),
    ))

    register_model!(ModelSpec(
        name="transit_chain_erlang",
        ode! = _transit_chain_erlang!,
        param_names=[:r, :K, :ktr, :emax, :ic50, :hill],
        bounds=[(1e-6, 5.0), (1e-3, 1e7), (1e-6, 10.0), (0.0, 10.0), (1e-8, 1e4), (0.1, 8.0)],
        n_states=4,
        observable=u -> u[1],
        base_growth_family="delay_surrogate",
        default_solver=Tsit5(),
        state_names=[:N, :X1, :X2, :X3],
        metadata=Dict(:family => :delay_surrogate),
    ))

    register_model!(ModelSpec(
        name="bi_exponential_response",
        ode! = _bi_exponential!,
        param_names=[:a, :b],
        bounds=[(1e-6, 10.0), (1e-6, 10.0)],
        n_states=2,
        observable=u -> u[1] + u[2],
        base_growth_family="phenomenological",
        default_solver=Tsit5(),
        state_names=[:N1, :N2],
        metadata=Dict(:family => :phenomenological),
    ))

    return nothing
end

end # module Registry
