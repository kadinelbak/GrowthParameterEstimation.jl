module Exposure

export AbstractExposure, ConstantExposure, PulseExposure, SteppedExposure, DecayingExposure,
       build_exposure, evaluate_exposure

"""
    AbstractExposure

Abstract type representing an exposure function over time.

All exposure types should subtype this abstract type and implement the
callable interface `(exposure::T)(t::Real) -> Float64` that returns the
exposure value at time `t`.
"""
abstract type AbstractExposure end

"""
    ConstantExposure(value::Float64)

An exposure that maintains a constant value over time.

# Arguments
- `value::Float64`: The constant exposure value

# Examples
```julia
# Create a constant exposure with value 0.5
exp = ConstantExposure(0.5)

# Evaluate at any time point (returns 0.5)
exp(0.0)    # returns 0.5
exp(10.0)   # returns 0.5
```
"""
struct ConstantExposure <: AbstractExposure
    value::Float64
end

(structure::ConstantExposure)(t::Real) = structure.value

"""
    PulseExposure(amplitude::Float64, start_time::Float64, end_time::Float64)

An exposure that delivers a constant amplitude during a specified time interval.

Outside the interval [start_time, end_time], the exposure is zero. Within the
interval, the exposure has the specified amplitude value.

# Arguments
- `amplitude::Float64`: The exposure value during the pulse interval
- `start_time::Float64`: The start time of the pulse (inclusive)
- `end_time::Float64`: The end time of the pulse (inclusive)

# Examples
```julia
# Create a pulse exposure with amplitude 2.0 from time 1.0 to 3.0
exp = PulseExposure(2.0, 1.0, 3.0)

# Evaluate at various time points
exp(0.0)   # returns 0.0 (before pulse)
exp(1.0)   # returns 2.0 (start of pulse)
exp(2.0)   # returns 2.0 (during pulse)
exp(3.0)   # returns 2.0 (end of pulse)
exp(4.0)   # returns 0.0 (after pulse)
```
"""
struct PulseExposure <: AbstractExposure
    amplitude::Float64
    start_time::Float64
    end_time::Float64
end

function (structure::PulseExposure)(t::Real)
    return (structure.start_time <= t <= structure.end_time) ? structure.amplitude : 0.0
end

"""
    SteppedExposure(change_times::Vector{Float64}, values::Vector{Float64})

An exposure that changes value at specified time points.

The exposure value remains constant between specified change times. At each
change time, the value steps to the corresponding entry in the values vector.
For times before the first change time, the first value is used. For times
after the last change time, the last value is used.

# Arguments
- `change_times::Vector{Float64}`: Times at which the exposure value changes
- `values::Vector{Float64}`: Corresponding exposure values for each interval

# Examples
```julia
# Create a stepped exposure that changes from 0.0 to 1.0 at time 2.0
exp = SteppedExposure([2.0], [0.0, 1.0])

# Evaluate at various time points
exp(0.0)   # returns 0.0 (before first change)
exp(1.0)   # returns 0.0 (before first change)
exp(2.0)   # returns 1.0 (at change time)
exp(3.0)   # returns 1.0 (after change)
```
"""
"""
    SteppedExposure(change_times::Vector{Float64}, values::Vector{Float64})

An exposure that changes value at specified time points.

The exposure value remains constant between specified change times. At each
change time, the value steps to the corresponding entry in the values vector.
For times before the first change time, the first value is used. For times
after the last change time, the last value is used.

# Arguments
- `change_times::Vector{Float64}`: Times at which the exposure value changes
- `values::Vector{Float64}`: Corresponding exposure values for each interval

# Examples
```julia
# Create a stepped exposure that changes from 0.0 to 1.0 at time 2.0
exp = SteppedExposure([2.0], [0.0, 1.0])

# Evaluate at various time points
exp(0.0)   # returns 0.0 (before first change)
exp(1.0)   # returns 0.0 (before first change)
exp(2.0)   # returns 1.0 (at change time)
exp(3.0)   # returns 1.0 (after change)
```
"""
struct SteppedExposure <: AbstractExposure
    change_times::Vector{Float64}
    values::Vector{Float64}
end

function (structure::SteppedExposure)(t::Real)
    idx = searchsortedlast(structure.change_times, Float64(t))
    idx = clamp(idx, 1, length(structure.values))
    return structure.values[idx]
end

"""
    DecayingExposure(c0::Float64, decay_rate::Float64, t0::Float64)

An exposure that follows an exponential decay profile starting at a specified time.

The exposure is zero before time t0, and follows the formula c0 * exp(-decay_rate * (t - t0))
for times t >= t0.

# Arguments
- `c0::Float64`: Initial exposure value at time t0
- `decay_rate::Float64`: Rate of decay (must be non-negative)
- `t0::Float64`: Start time of the decay process

# Examples
```julia
# Create a decaying exposure starting at time 1.0 with initial value 2.0 and decay rate 0.5
exp = DecayingExposure(2.0, 0.5, 1.0)

# Evaluate at various time points
exp(0.0)   # returns 0.0 (before decay starts)
exp(1.0)   # returns 2.0 (at start of decay)
exp(2.0)   # returns 2.0 * exp(-0.5 * 1.0) ≈ 1.21
exp(3.0)   # returns 2.0 * exp(-0.5 * 2.0) ≈ 0.74
```
"""
struct DecayingExposure <: AbstractExposure
    c0::Float64
    decay_rate::Float64
    t0::Float64
end

"""
    (exp::DecayingExposure)(t::Real)

Evaluate a decaying exposure at time t.

# Arguments
- `exp::DecayingExposure`: The decaying exposure to evaluate
- `t::Real`: The time point at which to evaluate the exposure

# Returns
- `Float64`: The exposure value at time t

# Examples
```julia
# Create a decaying exposure
exp = DecayingExposure(2.0, 0.5, 1.0)

# Evaluate at various time points
exp(0.0)   # returns 0.0 (before decay starts)
exp(1.0)   # returns 2.0 (at start of decay)
exp(2.0)   # returns 2.0 * exp(-0.5 * 1.0) ≈ 1.21
```
"""
function (structure::DecayingExposure)(t::Real)
    if t < structure.t0
        return 0.0
    end
    return structure.c0 * exp(-structure.decay_rate * (Float64(t) - structure.t0))
end

"""
    build_exposure(kind::Symbol; kwargs...)

Construct an exposure object of the specified type.

This function creates exposure objects based on the kind parameter and
additional keyword arguments specific to each exposure type.

# Arguments
- `kind::Symbol`: The type of exposure to create. Valid options are:
  - `:constant`: Creates a ConstantExposure (requires `:value`)
  - `:pulse`: Creates a PulseExposure (requires `:amplitude`, `:start_time`, `:end_time`)
  - `:stepped`: Creates a SteppedExposure (requires `:change_times`, `:values`)
  - `:decay`: Creates a DecayingExposure (requires `:c0`, `:decay_rate`, `:t0`)
- `kwargs...`: Keyword arguments specific to the exposure type being created

# Returns
- `AbstractExposure`: An exposure object of the specified type

# Examples
```julia
# Create a constant exposure
exp = build_exposure(:constant, value=0.5)

# Create a pulse exposure
exp = build_exposure(:pulse, amplitude=2.0, start_time=1.0, end_time=3.0)

# Create a stepped exposure
exp = build_exposure(:stepped, change_times=[0.0, 2.0, 5.0], values=[0.0, 1.0, 0.0])

# Create a decaying exposure
exp = build_exposure(:decay, c0=1.0, decay_rate=0.2, t0=0.0)
```
"""
function build_exposure(kind::Symbol; kwargs...)
    if kind == :constant
        return ConstantExposure(Float64(get(kwargs, :value, 0.0)))
    elseif kind == :pulse
        return PulseExposure(
            Float64(get(kwargs, :amplitude, 1.0)),
            Float64(get(kwargs, :start_time, 0.0)),
            Float64(get(kwargs, :end_time, 1.0)),
        )
    elseif kind == :stepped
        return SteppedExposure(
            Float64.(get(kwargs, :change_times, [0.0])),
            Float64.(get(kwargs, :values, [0.0])),
        )
    elseif kind == :decay
        return DecayingExposure(
            Float64(get(kwargs, :c0, 1.0)),
            Float64(get(kwargs, :decay_rate, 0.1)),
            Float64(get(kwargs, :t0, 0.0)),
        )
    else
        error("Unknown exposure kind: $kind")
    end
end

function evaluate_exposure(exposure::AbstractExposure, times::AbstractVector{<:Real})
    return [exposure(t) for t in times]
end

function evaluate_exposure(exposure::AbstractExposure, t::Real)
    return exposure(t)
end

end # module Exposure
