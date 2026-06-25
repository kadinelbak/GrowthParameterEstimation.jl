module Observation

export ObservationSpec, observed_signal, viable_total, sum_states

"""
    ObservationSpec

Specification for how to map model states to observable quantities.

# Fields
- `name::String`: Identifier for this observation specification
- `map_fn::Function`: Function that maps state vector to observable value (state, p, t) -> Real
- `scale::Float64`: Multiplicative scaling factor applied to the mapped value
- `sigma_add::Float64`: Additive noise standard deviation (for observation error model)
"""
struct ObservationSpec
    name::String
    map_fn::Function
    scale::Float64
    sigma_add::Float64
end

"""
    observed_signal(spec::ObservationSpec, state::AbstractVector, p, t)

Compute the observed signal from a model state using an observation specification.

This function applies the observation mapping function to the state and scales
the result according to the observation specification.

# Arguments
- `spec::ObservationSpec`: The observation specification to use
- `state::AbstractVector`: The current state of the model
- `p`: Parameters for the observation function
- `t`: Current time point

# Returns
- `Real`: The scaled observed signal value

# Examples
```julia
# Create an observation specification for direct measurement of first state
obs_spec = ObservationSpec("direct", (state, p, t) -> state[1], 1.0, 0.1)

# Compute observed signal for state [5.0, 2.0] at time 0.0 with no parameters
signal = observed_signal(obs_spec, [5.0, 2.0], (), 0.0)  # returns 5.0
```
"""
function observed_signal(spec::ObservationSpec, state::AbstractVector, p, t)
    return spec.scale * spec.map_fn(state, p, t)
end

"""
    viable_total(state::AbstractVector, p, t)

Return the viable total from a state vector.

By default, this returns the first state variable, representing the total
viable population or biomass. This function can be customized in observation
specifications to compute different viability measures.

# Arguments
- `state::AbstractVector`: The current state of the model
- `p`: Parameters for the observation function (unused in default implementation)
- `t`: Current time point (unused in default implementation)

# Returns
- `Real`: The viable total (first state variable by default)

# Examples
```julia
# For a state vector [100.0, 10.0] representing [viable, non-viable]
total = viable_total([100.0, 10.0], (), 0.0)  # returns 100.0
```
"""
viable_total(state::AbstractVector, p, t) = state[1]
"""
    sum_states(indices::Vector{Int})

Create a function that sums specific state variables from a state vector.

This function returns an anonymous function that computes the sum of specified
state variables from a state vector. Useful for creating observation specifications
that measure combined quantities (e.g., total population from multiple compartments).

# Arguments
- `indices::Vector{Int}`: Indices of state variables to sum (1-based indexing)

# Returns
- `Function`: A function with signature (state, p, t) -> Real that returns the sum
  of the specified state variables

# Examples
```julia
# Create an observation function that sums the first two state variables
sum_first_two = sum_states([1, 2])

# Use it to compute the sum of states [3.0, 4.0, 5.0]
result = sum_first_two([3.0, 4.0, 5.0], (), 0.0)  # returns 7.0

# Create an observation specification using this function
obs_spec = ObservationSpec("total_population", sum_states([1, 2]), 1.0, 0.1)
```
"""
sum_states(indices::Vector{Int}) = (state, p, t) -> sum(state[i] for i in indices)

end # module Observation
