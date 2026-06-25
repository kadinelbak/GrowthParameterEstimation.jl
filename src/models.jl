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
"""
    AbstractBaseModel

Abstract type for all base growth models.

All base models should subtype this abstract type and implement the
callable interface `(model::T)(u, p, t) -> Real` that returns the
growth rate du/dt given state u, parameters p, and time t.
"""
abstract type AbstractBaseModel end

"""
    AbstractModifier

Abstract type for all model modifiers.

All modifiers should subtype this abstract type and implement the
callable interface `(modifier::T)(base_du, u, p, t) -> Real` that returns
the modified growth rate given the base model output base_du, state u,
parameters p, and time t.
"""
abstract type AbstractModifier end

# ---------------------------------------------------------------------------
# Base model definitions (callable as (u, p, t) → du)
# ---------------------------------------------------------------------------
"""
    LogisticModel(r::Real, K::Real)

A logistic growth model.

The logistic growth model describes population growth that starts exponentially,
then slows as the population approaches the carrying capacity K due to
resource limitations.

# Arguments
- `r::Real`: The intrinsic growth rate
- `K::Real`: The carrying capacity (maximum population size)

# Examples
```julia
# Create a logistic growth model with growth rate 0.5 and carrying capacity 100.0
model = LogisticModel(0.5, 100.0)

# Evaluate the growth rate at population size 50.0
growth_rate = model(50.0, (), 0.0)  # returns 0.5 * 50.0 * (1 - 50.0/100.0) = 12.5
```
"""
struct LogisticModel <: AbstractBaseModel
    r::Real
    K::Real
    end
(m::LogisticModel)(u, p, t) = m.r * u * (1 - u / m.K)
"""
    GompertzModel(a::Real, b::Real, K::Real)

A Gompertz growth model.

The Gompertz model describes growth that is slowest at the beginning and end of a time period.
The parameter `b` is retained for API compatibility but is not used in the simple form.

# Arguments
- `a::Real`: The growth rate parameter
- `b::Real`: A parameter retained for API compatibility (not used in calculation)
- `K::Real`: The carrying capacity (maximum population size)

# Examples
```julia
# Create a Gompertz growth model
model = GompertzModel(0.5, 1.0, 100.0)

# Evaluate the growth rate at population size 50.0
# Note: For u=50.0 and K=100.0, log(K/u) = log(2) ≈ 0.693
growth_rate = model(50.0, (), 0.0)  # returns 0.5 * 50.0 * log(100.0/50.0) ≈ 17.33
```
"""
struct GompertzModel <: AbstractBaseModel
    a::Real
    b::Real   # retained for API compatibility; not used in the simple form
    K::Real
        end
function (m::GompertzModel)(u, p, t)
    u <= 0 || u ≥ m.K ? 0.0 : m.a * u * log(m.K / u)
    end
"""
    ExponentialModel(r::Real)

An exponential growth model.

The exponential growth model describes population growth at a constant rate.
This model assumes unlimited resources and constant growth conditions.

# Arguments
- `r::Real`: The growth rate constant

# Examples
```julia
# Create an exponential growth model with growth rate 0.3
model = ExponentialModel(0.3)

# Evaluate the growth rate at population size 50.0
growth_rate = model(50.0, (), 0.0)  # returns 0.3 * 50.0 = 15.0
```
"""
struct ExponentialModel <: AbstractBaseModel
    r::Real
    end
(m::ExponentialModel)(u, p, t) = m.r * u
"""
    LogisticWithDeathModel(r::Real, K::Real, death_rate::Real)

A logistic growth model with death rate.

This model extends the logistic growth model by adding a linear death term.
It describes population growth that is limited by both carrying capacity and
constant death rate.

# Arguments
- `r::Real`: The intrinsic growth rate
- `K::Real`: The carrying capacity (maximum population size)
- `death_rate::Real`: The constant death rate

# Examples
```julia
# Create a logistic growth model with death rate
model = LogisticWithDeathModel(0.5, 100.0, 0.05)

# Evaluate the growth rate at population size 50.0
# Logistic component: 0.5 * 50.0 * (1 - 50.0/100.0) = 12.5
# Death component: 0.05 * 50.0 = 2.5
# Net growth rate: 12.5 - 2.5 = 10.0
growth_rate = model(50.0, (), 0.0)  # returns 10.0
```
"""
struct LogisticWithDeathModel <: AbstractBaseModel
    r::Real
    K::Real
    death_rate::Real
    end
(m::LogisticWithDeathModel)(u, p, t) = m.r * u * (1 - u / m.K) - m.death_rate * u
"""
    GompertzWithDeathModel(a::Real, b::Real, K::Real, death_rate::Real)

A Gompertz growth model with death rate.

This model extends the Gompertz growth model by adding a linear death term.
The parameter `b` is retained for API compatibility but is not used in the simple form.

# Arguments
- `a::Real`: The growth rate parameter
- `b::Real`: A parameter retained for API compatibility (not used in calculation)
- `K::Real`: The carrying capacity (maximum population size)
- `death_rate::Real`: The constant death rate

# Examples
```julia
# Create a Gompertz growth model with death rate
model = GompertzWithDeathModel(0.5, 1.0, 100.0, 0.05)

# Evaluate the growth rate at population size 50.0
# For u=50.0 and K=100.0, log(K/u) = log(2) ≈ 0.693
# Gompertz component: 0.5 * 50.0 * log(100.0/50.0) ≈ 17.33
# Death component: 0.05 * 50.0 = 2.5
# Net growth rate: 17.33 - 2.5 = 14.83
growth_rate = model(50.0, (), 0.0)  # returns approximately 14.83
```
"""
"""
    GompertzWithDeathModel(a::Real, b::Real, K::Real, death_rate::Real)

A Gompertz growth model with death rate.

This model extends the Gompertz growth model by adding a linear death term.
The parameter `b` is retained for API compatibility but is not used in the simple form.

# Arguments
- `a::Real`: The growth rate parameter
- `b::Real`: A parameter retained for API compatibility (not used in calculation)
- `K::Real`: The carrying capacity (maximum population size)
- `death_rate::Real`: The constant death rate

# Examples
```julia
# Create a Gompertz growth model with death rate
model = GompertzWithDeathModel(0.5, 1.0, 100.0, 0.05)

# Evaluate the growth rate at population size 50.0
# For u=50.0 and K=100.0, log(K/u) = log(2) ≈ 0.693
# Gompertz component: 0.5 * 50.0 * log(100.0/50.0) ≈ 17.33
# Death component: 0.05 * 50.0 = 2.5
# Net growth rate: 17.33 - 2.5 = 14.83
growth_rate = model(50.0, (), 0.0)  # returns approximately 14.83
```
"""
struct GompertzWithDeathModel <: AbstractBaseModel
    a::Real
    b::Real
    K::Real
    death_rate::Real
        end
function (m::GompertzWithDeathModel)(u, p, t)
    u <= 0 || u ≥ m.K ? -m.death_rate * u : m.a * u * log(m.K / u) - m.death_rate * u
    end
"""
    ExponentialWithDelayModel(r::Real, K::Real, tlag::Real)

An exponential growth model with delay.

This model extends the exponential growth model by adding a time lag (tlag) before
growth begins. For times less than tlag, the growth rate is zero. For times
greater than or equal to tlag, growth proceeds exponentially at rate r.

Note: There appears to be an error in the original implementation - the growth
equation includes a carrying capacity term (1 - u/K) which is not typical for
pure exponential growth. This may be intentional for specific use cases.

# Arguments
- `r::Real`: The growth rate constant (active after tlag)
- `K::Real`: A carrying capacity parameter (unusual for exponential model)
- `tlag::Real`: The time lag before growth begins

# Examples
```julia
# Create an exponential growth model with delay
model = ExponentialWithDelayModel(0.5, 100.0, 2.0)

# Evaluate the growth rate at different times and population sizes
# Before delay period (t=1.0 < tlag=2.0): growth rate is 0
growth_rate1 = model(50.0, (), 1.0)  # returns 0.0

# After delay period (t=3.0 >= tlag=2.0): 
# Note: Includes carrying capacity term (1 - u/K)
growth_rate2 = model(50.0, (), 3.0)  # returns 0.5 * 50.0 * (1 - 50.0/100.0) = 12.5
```
"""
"""
    ExponentialWithDelayModel(r::Real, K::Real, tlag::Real)

An exponential growth model with delay.

This model extends the exponential growth model by adding a time lag (tlag) before
growth begins. For times less than tlag, the growth rate is zero. For times
greater than or equal to tlag, growth proceeds exponentially at rate r.

Note: There appears to be an error in the original implementation - the growth
equation includes a carrying capacity term (1 - u/K) which is not typical for
pure exponential growth. This may be intentional for specific use cases.

# Arguments
- `r::Real`: The growth rate constant (active after tlag)
- `K::Real`: A carrying capacity parameter (unusual for exponential model)
- `tlag::Real`: The time lag before growth begins

# Examples
```julia
# Create an exponential growth model with delay
model = ExponentialWithDelayModel(0.5, 100.0, 2.0)

# Evaluate the growth rate at different times and population sizes
# Before delay period (t=1.0 < tlag=2.0): growth rate is 0
growth_rate1 = model(50.0, (), 1.0)  # returns 0.0

# After delay period (t=3.0 >= tlag=2.0): 
# Note: Includes carrying capacity term (1 - u/K)
growth_rate2 = model(50.0, (), 3.0)  # returns 0.5 * 50.0 * (1 - 50.0/100.0) = 12.5
```
"""
struct ExponentialWithDelayModel <: AbstractBaseModel
    r::Real
    K::Real
    tlag::Real
    end
(m::ExponentialWithDelayModel)(u, p, t) = (t >= m.tlag ? m.r : 0.0) * u * (1 - u / m.K)
"""
    LogisticWithDelayModel(r::Real, K::Real, tlag::Real)

A logistic growth model with delay.

This model extends the logistic growth model by adding a time lag (tlag) before
growth begins. For times less than tlag, the growth rate is zero. For times
greater than or equal to tlag, growth proceeds logistically at rate r with
carrying capacity K.

# Arguments
- `r::Real`: The intrinsic growth rate (active after tlag)
- `K::Real`: The carrying capacity (maximum population size)
- `tlag::Real`: The time lag before growth begins

# Examples
```julia
# Create a logistic growth model with delay
model = LogisticWithDelayModel(0.5, 100.0, 2.0)

# Evaluate the growth rate at different times and population sizes
# Before delay period (t=1.0 < tlag=2.0): growth rate is 0
growth_rate1 = model(50.0, (), 1.0)  # returns 0.0

# After delay period (t=3.0 >= tlag=2.0): 
growth_rate2 = model(50.0, (), 3.0)  # returns 0.5 * 50.0 * (1 - 50.0/100.0) = 12.5
```
"""
struct LogisticWithDelayModel <: AbstractBaseModel
    r::Real
    K::Real
    tlag::Real
    end
(m::LogisticWithDelayModel)(u, p, t) = (t >= m.tlag ? m.r : 0.0) * u * (1 - u / m.K)
"""
    ExponentialWithDeathAndDelayModel(r::Real, K::Real, death_rate::Real, tlag::Real)

An exponential growth model with death rate and delay.

This model combines exponential growth with both a death rate and a time lag.
For times less than tlag, the growth rate is zero. For times greater than or
equal to tlag, growth proceeds exponentially at rate r minus a linear death term.

Note: Similar to ExponentialWithDelayModel, this implementation includes a
carrying capacity term (1 - u/K) which is unusual for exponential models but
may be intentional for specific use cases.

# Arguments
- `r::Real`: The growth rate constant (active after tlag)
- `K::Real`: A carrying capacity parameter (unusual for exponential model)
- `death_rate::Real`: The constant death rate
- `tlag::Real`: The time lag before growth begins

# Examples
```julia
# Create an exponential growth model with death and delay
model = ExponentialWithDeathAndDelayModel(0.5, 100.0, 0.05, 2.0)

# Evaluate the growth rate at different times and population sizes
# Before delay period (t=1.0 < tlag=2.0): growth rate is 0
growth_rate1 = model(50.0, (), 1.0)  # returns 0.0

# After delay period (t=3.0 >= tlag=2.0): 
# Exponential component: 0.5 * 50.0 * (1 - 50.0/100.0) = 12.5
# Death component: 0.05 * 50.0 = 2.5
# Net growth rate: 12.5 - 2.5 = 10.0
growth_rate2 = model(50.0, (), 3.0)  # returns 10.0
```
"""
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
"""
    create_model(::Type{T}, params) where T

Create a model instance from a parameter vector.

This function provides a convenient way to instantiate model types using
parameter vectors, which is particularly useful for optimization routines.

# Arguments
- `::Type{T}`: The model type to create
- `params`: A vector or tuple of parameters for the model

# Returns
- `T`: An instance of the specified model type

# Examples
```julia
# Create a logistic model from parameters [growth_rate, carrying_capacity]
params = [0.5, 100.0]
model = create_model(LogisticModel, params)

# Create a Gompertz model from parameters [growth_rate, unused, carrying_capacity]
params = [0.5, 1.0, 100.0]
model = create_model(GompertzModel, params)
```
"""

# ---------------------------------------------------------------------------
# Modifier definitions
# ---------------------------------------------------------------------------
"""
    DeathModifier(death_rate::Real)

A modifier that adds a linear death term to a base growth model.

This modifier subtracts a death term proportional to the population size from
the base growth rate, representing constant mortality or death processes.

# Arguments
- `death_rate::Real`: The constant death rate (must be non-negative)

# Examples
```julia
# Create a death modifier with rate 0.05
modifier = DeathModifier(0.05)

# Apply to a base growth rate of 10.0 at population size 50.0
# Death component: 0.05 * 50.0 = 2.5
# Modified growth rate: 10.0 - 2.5 = 7.5
modified_rate = modifier(10.0, 50.0, (), 0.0)  # returns 7.5
```
"""
struct DeathModifier <: AbstractModifier
    death_rate::Real
end
(m::DeathModifier)(base_du, u, p, t) = base_du - m.death_rate * u

"""
    LagPhaseModifier(tlag::Real)

A modifier that implements a lag phase delay in growth.

This modifier sets the growth rate to zero for times less than the lag period,
and allows unrestricted growth (returns base growth rate unchanged) for times
greater than or equal to the lag period.

# Arguments
- `tlag::Real`: The lag time period (must be non-negative)

# Examples
```julia
# Create a lag phase modifier with 2.0 time units lag
modifier = LagPhaseModifier(2.0)

# Apply to a base growth rate of 5.0 at time 1.0 (before lag period)
# Returns 0.0 (growth is suppressed during lag phase)
modified_rate1 = modifier(5.0, (), (), 1.0)  # returns 0.0

# Apply to a base growth rate of 5.0 at time 3.0 (after lag period)
# Returns 5.0 (unrestricted growth after lag period)
modified_rate2 = modifier(5.0, (), (), 3.0)  # returns 5.0
```
"""
struct LagPhaseModifier <: AbstractModifier
    tlag::Real
end
(m::LagPhaseModifier)(base_du, u, p, t) = t >= m.tlag ? base_du : zero(base_du)

"""
    HillInhibitionModifier(emax::Real, ic50::Real, hill::Real)

A modifier that implements Hill-type inhibition of growth.

This modifier reduces the growth rate by a factor that depends on drug concentration
according to the Hill equation. It represents inhibitory effects such as drug-induced
growth suppression.

# Arguments
- `emax::Real`: The maximum inhibitory effect (between 0 and 1, where 1 is complete inhibition)
- `ic50::Real`: The drug concentration at which the inhibitory effect is half of emax
- `hill::Real`: The Hill coefficient, which determines the steepness of the dose-response curve

# Examples
```julia
# Create a Hill inhibition modifier with emax=0.8, ic50=10.0, hill=2.0
modifier = HillInhibitionModifier(0.8, 10.0, 2.0)

# Apply to a base growth rate of 5.0 with drug concentration 5.0
# Inhibition calculation: 
#   ic50_safe = max(10.0, eps) = 10.0
#   inhibition = 0.8 * 5.0^2 / (10.0^2 + 5.0^2 + eps) ≈ 0.8 * 25 / (100 + 25) = 0.16
#   inhibition factor = max(0.0, 1.0 - 0.16) = 0.84
#   modified growth rate = 5.0 * 0.84 = 4.2
modified_rate = modifier(5.0, (), (:drug => 5.0), 0.0)  # returns approximately 4.2
```
"""
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

"""
    HillKillModifier(emax_kill::Real, ic50::Real, hill::Real)

A modifier that implements Hill-type killing of growth.

This modifier subtracts a kill term from the growth rate that depends on drug concentration
according to the Hill equation. It represents cytotoxic effects such as drug-induced cell death.

# Arguments
- `emax_kill::Real`: The maximum killing effect (typically >= 0)
- `ic50::Real`: The drug concentration at which the killing effect is half of emax_kill
- `hill::Real`: The Hill coefficient, which determines the steepness of the dose-response curve

# Examples
```julia
# Create a Hill kill modifier with emax_kill=0.5, ic50=10.0, hill=2.0
modifier = HillKillModifier(0.5, 10.0, 2.0)

# Apply to a base growth rate of 5.0 with drug concentration 5.0 and population 50.0
# Kill calculation: 
#   ic50_safe = max(10.0, eps) = 10.0
#   kill = 0.5 * 5.0^2 / (10.0^2 + 5.0^2 + eps) ≈ 0.5 * 25 / (100 + 25) = 0.1
#   kill component = kill * u = 0.1 * 50.0 = 5.0
#   modified growth rate = 5.0 - 5.0 = 0.0
modified_rate = modifier(5.0, (), (:drug => 5.0), 50.0)  # returns 0.0
```
"""
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

