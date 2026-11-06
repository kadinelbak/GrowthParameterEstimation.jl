# Plan for Buildable Models in GrowthParameterEstimation.jl

## Current State Analysis

The package already has a strong foundation for composable models:
- Abstract types: `AbstractBaseModel`, `AbstractModifier`
- Base models: LogisticModel, GompertzModel, ExponentialModel, etc.
- Modifiers: DeathModifier, LagPhaseModifier, HillInhibitionModifier, HillKillModifier
- Composition: `CompositeModel` combines base + modifier
- Registration: `ModelSpec` system in Registry module
- ODE conversion: `to_ode!` function converts models to DifferentialEquations.jl format

## Requirements for Cancer Growth Models

Based on the request, we need to support:
1. **Multi-compartment models** (multiple ODEs)
2. **Multiple populations** (e.g., sensitive/resistant cells)
3. **Multiple effectors** (drugs, immune responses, etc.)
4. **Allele-specific terms** (genetic variants)
5. **Theta terms** (shape parameters in growth equations)
6. **Death terms** (cell death rates)
7. **Similar biological modifiers**

## Implementation Plan

### Phase 1: Enhanced Builder API (Non-breaking)

Add to `src/models.jl`:

#### 1.1 Export Builder Functions
```julia
# In Models module export block
export
    # ... existing exports ...
    # Builder functions
    build_logistic, build_gompertz, build_exponential,
    apply_death, apply_lag, apply_hill_inhibition, apply_hill_kill,
    compose_models
```

#### 1.2 Convenience Builder Functions
```julia
# Simplified model creation
function build_logistic(; r=1.0, K=1.0)
    LogisticModel(r, K)
end

function build_gompertz(; a=1.0, b=1.0, K=1.0)
    GompertzModel(a, b, K)
end

# Modifier application functions
function apply_death(model::AbstractBaseModel; death_rate=0.0)
    apply_modifier(model, DeathModifier; death_rate)
end

function apply_lag(model::AbstractBaseModel; tlag=0.0)
    apply_modifier(model, LagPhaseModifier; tlag)
end
```

#### 1.3 Multi-composition Support
```julia
# Allow composing multiple modifiers
function compose_models(base::AbstractBaseModel, modifiers::Vector{Type{<:AbstractModifier}}; 
                       kwargs...)
    model = base
    for modifier_type in modifiers
        # Extract relevant kwargs for this modifier
        mod_params = filter(kwargs) do (k, v)
            hasfield(modifier_type, k)
        end
        model = apply_modifier(model, modifier_type; mod_params...)
    end
    return model
end
```

### Phase 2: Enhanced Registration System

Add to `src/registry.jl`:

#### 2.1 Smart ModelSpec Constructors
```julia
# In Registry module
function ModelSpec(; name::AbstractString,
                   model::AbstractBaseModel,
                   bounds=nothing,
                   observable=u -> u[1],
                   default_solver=Tsit5(),
                   kwargs...)
    # Auto-extract parameter names from model
    param_names = _extract_param_names(model)
    
    # Auto-determine n_states from model
    n_states = _detect_n_states(model)
    
    # Provide sensible defaults for bounds if not given
    if isnothing(bounds)
        bounds = _suggest_default_bounds(model)
    end
    
    # Convert model to ode! function
    ode_func = Models.to_ode!(model)
    
    ModelSpec(; name, ode! = ode_func, param_names, bounds, n_states,
              observable, base_growth_family = get(kwargs, :family, "custom"),
              default_solver, p0_factory = nothing,
              fixed_params = Dict{Int,Float64}(),
              state_names = _default_state_names(n_states),
              metadata = Dict(kwargs))
end
```

#### 2.2 Helper Functions
```julia
function _extract_param_names(model::AbstractBaseModel)
    fieldnames(typeof(model))
end

function _detect_n_states(model::AbstractBaseModel)
    # Default to 1, can be overridden for specific model types
    return 1
end

function _suggest_default_bounds(model::AbstractBaseModel)
    # Provide reasonable defaults based on model type
    params = _extract_param_names(model)
    bounds = Tuple{Float64,Float64}[]
    for param in params
        if string(param) in ("r", "a")  # growth rates
            push!(bounds, (1e-6, 5.0))
        elseif string(param) in ("K",)   # carrying capacity
            push!(bounds, (1e-3, 1e7))
        elseif string(param) in ("death_rate", "emax", "kill_coeff")  # rates
            push!(bounds, (0.0, 20.0))
        elseif string(param) in ("tlag", "hill")  # time/hill coeff
            push!(bounds, (0.0, 10.0))
        elseif string(param) in ("ic50", "KR", "KS")  # concentrations
            push!(bounds, (1e-8, 1e4))
        else
            push!(bounds, (1e-6, 10.0))  # default
        end
    end
    return bounds
end
```

#### 2.3 Registration Convenience Functions
```julia
function register_composable_model(name::String, model::AbstractBaseModel;
                                  bounds=nothing, observable=u->u[1],
                                  default_solver=Tsit5(), kwargs...)
    spec = ModelSpec(; name, model, bounds, observable, default_solver, kwargs...)
    register_model!(spec)
    return spec
end
```

### Phase 3: Specialized Model Builders for Cancer Growth

Add specialized builders for common cancer model patterns:

#### 3.1 Population Structure Builders
```julia
# Two-population builder (sensitive/resistant)
function build_two_population(base_model_constructor::Function;
                             sensitive_params=Dict(),
                             resistant_params=Dict(),
                             mutation_rate=0.0,
                             death_rates=Dict(:S=>0.0, :R=>0.0))
    # Returns a function that creates the full ODE system
    function model_constructor(params...)
        # Split params for S and R populations
        # Construct base models for each
        # Return coupled ODE system
    end
    return model_constructor
end

# Allele-specific builder
function build_allele_specific(base_model_constructor::Function;
                              alleles::Vector{String},
                              fitness_effects=Dict())
    # Creates model with separate compartments for each allele
end
```

#### 3.2 Effector System Builders
```julia
# Drug effector builder
function build_drug_effector(base_model::AbstractBaseModel;
                            drug_effect_type=:inhibition,  # or :killing
                            ec50=1.0, hill=1.0, emax=1.0)
    # Adds drug effect terms to the model
end

# Immune effector builder
function build_immune_effector(base_model::AbstractBaseModel;
                              immune_cell_type=:cytotoxic,
                              kill_rate=0.1, half_sat=10.0)
    # Adds immune-mediated killing terms
end
```

### Phase 4: Usage Examples

After implementation, users should be able to do:

```julia
using GrowthParameterEstimation

# Build a logistic growth model with death and lag phase
base_model = build_logistic(r=0.5, K=1e9)
model_with_death = apply_death(base_model; death_rate=0.1)
final_model = apply_lag(model_with_death; tlag=2.0)

# Or using composition
final_model = compose_models(
    build_logistic(r=0.5, K=1e9),
    [DeathModifier, LagPhaseModifier];
    death_rate=0.1, tlag=2.0
)

# Register for use in fitting/pipeline
register_composable_model("my_cancer_model", final_model;
                         bounds=[(1e-6, 2.0), (1e6, 1e10), (0.0, 1.0), (0.0, 5.0)],
                         family="cancer")

# For two-population model
two_pop_model = build_two_population(
    build_logistic;
    sensitive_params=Dict(:r=>0.6, :K=>1e9),
    resistant_params=Dict(:r=>0.3, :K=>5e8),
    mutation_rate=1e-6
)

# With drug effects
drug_model = build_drug_effector(two_pop_model;
                                drug_effect_type=:killing,
                                ec50=0.5, hill=2.0, emax=0.8)
```

## Backward Compatibility

All existing code will continue to work:
- Existing model definitions unchanged
- Legacy ODE RHS functions preserved
- Registration system unchanged
- All exported functions maintain same signatures

## Files to Modify

1. `src/models.jl` - Add builder functions and exports
2. `src/registry.jl` - Add smart constructors and helpers
3. Update documentation in README.md and docstrings

## Next Steps

1. Implement the builder functions in models.jl
2. Implement the enhanced registration helpers in registry.jl
3. Add comprehensive docstrings and examples
4. Test with existing models to ensure no regressions
5. Create example usage scripts demonstrating new capabilities