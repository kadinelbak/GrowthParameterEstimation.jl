# Models module - Contains all ODE model definitions
module Models

export
    # Composable API
    AbstractBaseModel, AbstractModifier,
       LogisticModel, GompertzModel, ExponentialModel,
       LogisticWithDeathModel, GompertzWithDeathModel,
       ExponentialWithDelayModel, LogisticWithDelayModel,
       ExponentialWithDeathAndDelayModel,
       DeathModifier, LagPhaseModifier, HillInhibitionModifier, HillKillModifier,

    CompositeModel, apply_modifier, create_model, from_params, to_ode!,
    # Builder functions for easy model construction
    build_logistic, build_gompertz, build_exponential,
    apply_death, apply_lag, apply_hill_inhibition, apply_hill_kill,
    compose_models

# ---------------------------------------------------------------------------
# Abstract types
# ---------------------------------------------------------------------------
abstract type AbstractBaseModel end
abstract type AbstractModifier end

# ---------------------------------------------------------------------------
# Base model definitions (callable as (u, p, t) → du)
# ---------------------------------------------------------------------------
struct LogisticModel <: AbstractBaseModel
    r::Real
    K::Real
    end
(m::LogisticModel)(u, p, t) = m.r * u * (1 - u / m.K)
struct GompertzModel <: AbstractBaseModel
    a::Real
    b::Real   # retained for API compatibility; not used in the simple form
    K::Real
        end
function (m::GompertzModel)(u, p, t)
    u <= 0 || u ≥ m.K ? 0.0 : m.a * u * log(m.K / u)
    end
struct ExponentialModel <: AbstractBaseModel
    r::Real
    end
(m::ExponentialModel)(u, p, t) = m.r * u
struct LogisticWithDeathModel <: AbstractBaseModel
    r::Real
    K::Real
    death_rate::Real
    end
(m::LogisticWithDeathModel)(u, p, t) = m.r * u * (1 - u / m.K) - m.death_rate * u
struct GompertzWithDeathModel <: AbstractBaseModel
    a::Real
    b::Real
    K::Real
    death_rate::Real
        end
function (m::GompertzWithDeathModel)(u, p, t)
    u <= 0 || u ≥ m.K ? -m.death_rate * u : m.a * u * log(m.K / u) - m.death_rate * u
    end
struct ExponentialWithDelayModel <: AbstractBaseModel
    r::Real
    K::Real
    tlag::Real
    end
(m::ExponentialWithDelayModel)(u, p, t) = (t >= m.tlag ? m.r : 0.0) * u * (1 - u / m.K)
struct LogisticWithDelayModel <: AbstractBaseModel
    r::Real
    K::Real
    tlag::Real
    end
(m::LogisticWithDelayModel)(u, p, t) = (t >= m.tlag ? m.r : 0.0) * u * (1 - u / m.K)
struct ExponentialWithDeathAndDelayModel <: AbstractBaseModel
    r::Real
    K::Real
    death_rate::Real
    tlag::Real
    end
(m::ExponentialWithDeathAndDelayModel)(u, p, t) =
    (t >= m.tlag ? m.r : 0.0) * u * (1 - u / m.K) - m.death_rate * u

# ---------------------------------------------------------------------------
# Factory for base models (convenient construction from parameter vectors)
# ---------------------------------------------------------------------------
create_model(::Type{LogisticModel}, params) = LogisticModel(params[1], params[2])
create_model(::Type{GompertzModel}, params) = GompertzModel(params[1], params[2], params[3])
create_model(::Type{ExponentialModel}, params) = ExponentialModel(params[1])
create_model(::Type{LogisticWithDeathModel}, params) = LogisticWithDeathModel(params[1], params[2], params[3])
create_model(::Type{GompertzWithDeathModel}, params) = GompertzWithDeathModel(params[1], params[2], params[3], params[4])
create_model(::Type{ExponentialWithDelayModel}, params) = ExponentialWithDelayModel(params[1], params[2], params[3])
create_model(::Type{LogisticWithDelayModel}, params) = LogisticWithDelayModel(params[1], params[2], params[3])
create_model(::Type{ExponentialWithDeathAndDelayModel}, params) = ExponentialWithDeathAndDelayModel(params[1], params[2], params[3], params[4])

# ---------------------------------------------------------------------------
# Modifier definitions
# ---------------------------------------------------------------------------
struct DeathModifier <: AbstractModifier
    death_rate::Real
end
(m::DeathModifier)(base_du, u, p, t) = base_du - m.death_rate * u

struct LagPhaseModifier <: AbstractModifier
    tlag::Real
end
(m::LagPhaseModifier)(base_du, u, p, t) = t >= m.tlag ? base_du : zero(base_du)

struct HillInhibitionModifier <: AbstractModifier
    emax::Real
    ic50::Real
    hill::Real
end
function (m::HillInhibitionModifier)(base_du, u, p, t)
    emax = m.emax
    ic50 = m.ic50
    hill = m.hill
    drug = max(get(p, :drug, 0.0), 0.0)
    ic50_safe = max(ic50, eps(eltype(base_du)))
    inhibition = emax * drug^hill / (ic50_safe^hill + drug^hill + eps(eltype(base_du)))
    return base_du * max(0.0, 1.0 - inhibition)
end

struct HillKillModifier <: AbstractModifier
    emax_kill::Real
    ic50::Real
    hill::Real
end
function (m::HillKillModifier)(base_du, u, p, t)
    emax_kill = m.emax_kill
    ic50 = m.ic50
    hill = m.hill
    drug = max(get(p, :drug, 0.0), 0.0)
    ic50_safe = max(ic50, eps(eltype(base_du)))
    kill = emax_kill * drug^hill / (ic50_safe^hill + drug^hill + eps(eltype(base_du)))
    return base_du - kill * u
end

# ---------------------------------------------------------------------------
# Composite model – combines a base model with a single modifier
# ---------------------------------------------------------------------------
struct CompositeModel <: AbstractBaseModel
    base::AbstractBaseModel
    modifier::AbstractModifier
end
function (c::CompositeModel)(u, p, t)
    base_du = c.base(u, p, t)
    return c.modifier(base_du, u, p, t)
end

# ---------------------------------------------------------------------------
# Helper to fetch a parameter from a Dict/NamedTuple/positional container.
# ---------------------------------------------------------------------------
_get_param(params, key::Symbol, idx::Int) =
    params isa AbstractDict ? params[key] :
    params isa NamedTuple ? getfield(params, key) :
    params[idx]

# --------------------------------------------------------------------------- 
# Modifier construction from generic parameter containers
# ---------------------------------------------------------------------------
from_params(::Type{DeathModifier}, params) = DeathModifier(_get_param(params, :death_rate, 1))
from_params(::Type{LagPhaseModifier}, params) = LagPhaseModifier(_get_param(params, :tlag, 1))
from_params(::Type{HillInhibitionModifier}, params) = HillInhibitionModifier(
        _get_param(params, :emax, 1),
        _get_param(params, :ic50, 2),
    _get_param(params, :hill, 3)
    )
from_params(::Type{HillKillModifier}, params) = HillKillModifier(
        _get_param(params, :emax_kill, 1),
    _get_param(params, :ic50, 2),
    _get_param(params, :hill, 3)
    )

# ---------------------------------------------------------------------------
# Apply a modifier to a base model, returning a CompositeModel
# ---------------------------------------------------------------------------
apply_modifier(model::AbstractBaseModel, ::Type{T}, params) where T<:AbstractModifier =
    CompositeModel(model, from_params(T, params))

# ---------------------------------------------------------------------------
# Convert a composable model to a DifferentialEquations.jl‑compatible in‑place ODE function
# ---------------------------------------------------------------------------
function to_ode!(model::AbstractBaseModel)
    return function (du, u, p, t)
        du[1] = model(u[1], p, t)
        return nothing
    end
end



# ---------------------------------------------------------------------------
# Builder functions for easy model construction
# ---------------------------------------------------------------------------
"""
    build_logistic(; r=1.0, K=1.0)

Create a LogisticModel with given parameters.
"""
function build_logistic(; r::Real = 1.0, K::Real = 1.0)
    return LogisticModel(r, K)
end

"""
    build_gompertz(; a=1.0, b=1.0, K=1.0)

Create a GompertzModel with given parameters.
"""
function build_gompertz(; a::Real = 1.0, b::Real = 1.0, K::Real = 1.0)
    return GompertzModel(a, b, K)
end

"""
    build_exponential(; r=1.0)

Create an ExponentialModel with given growth rate.
"""
function build_exponential(; r::Real = 1.0)
    return ExponentialModel(r)
end

"""
    apply_death(model::AbstractBaseModel; death_rate=0.0)

Apply a death modifier to a base model.
"""
function apply_death(model::AbstractBaseModel; death_rate::Real = 0.0)
    return apply_modifier(model, DeathModifier, (death_rate,))
end

"""
    apply_lag(model::AbstractBaseModel; tlag=0.0)

Apply a lag phase modifier to a base model.
"""
function apply_lag(model::AbstractBaseModel; tlag::Real = 0.0)
    return apply_modifier(model, LagPhaseModifier, (tlag,))
end

"""
    apply_hill_inhibition(model::AbstractBaseModel; emax=1.0, ic50=1.0, hill=1.0)

Apply a Hill inhibition modifier to a base model.
"""
function apply_hill_inhibition(model::AbstractBaseModel; 
                               emax::Real = 1.0, 
                               ic50::Real = 1.0, 
                               hill::Real = 1.0)
    return apply_modifier(model, HillInhibitionModifier, (emax, ic50, hill))
end

"""
    apply_hill_kill(model::AbstractBaseModel; emax_kill=1.0, ic50=1.0, hill=1.0)

Apply a Hill killing modifier to a base model.
"""
function apply_hill_kill(model::AbstractBaseModel; 
                        emax_kill::Real = 1.0, 
                        ic50::Real = 1.0, 
                        hill::Real = 1.0)
    return apply_modifier(model, HillKillModifier, (emax_kill, ic50, hill))
end

"""
    compose_models(base::AbstractBaseModel, modifiers::Vector{Type{<:AbstractModifier}}; 
                   kwargs...)

Compose a base model with multiple modifiers.
"""
function compose_models(base::AbstractBaseModel, 
                       modifiers::Vector{Type{<:AbstractModifier}}; 
                       kwargs...)
    model = base
    for modifier_type in modifiers
        # Extract parameters relevant to this modifier
        mod_params = Dict{Symbol,Any}()
        for (key, value) in kwargs
            if hasfield(modifier_type, key)
                mod_params[key] = value
            end
        end
        model = apply_modifier(model, modifier_type; mod_params...)
    end
    return model
end

end # module Models

