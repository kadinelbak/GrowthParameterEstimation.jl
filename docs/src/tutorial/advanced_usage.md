# Advanced Usage

This tutorial covers advanced features and workflows in GrowthParameterEstimation.jl.

## Staged Fitting Workflows

The package includes a sophisticated workflow system for staged model fitting:

```julia
using GrowthParameterEstimation.Workflow

# Create a custom workflow configuration
config = PipelineConfig(
    "1.0.0",
    ["logistic_growth", "gompertz_growth"],  # models to consider
    10,                                       # n_starts
    3,                                        # top_k
    500,                                      # maxiters
    1e-8,                                     # reltol
    1e-8,                                     # abstol
    true,                                     # weighted
    42,                                       # seed
    "advanced_results"                        # output_dir
)

# Define custom pipeline stages
stages = [
    PipelineStage(
        "initial_fit",
        "Initial model fitting to identify best candidates",
        (row) -> true,  # accept all rows
        [:cell_line, :treatment],             # condition columns
        ["logistic_growth", "gompertz_growth"], # models to test
        [:r],                                 # shared parameters (growth rate)
        Dict{Symbol,Float64}(),               # no fixed parameters
        Dict{Symbol,Tuple{String,Symbol}}()   # no inherited parameters
    ),
    PipelineStage(
        "refined_fit",
        "Refined fitting with parameter inheritance",
        (row) -> true,
        [:cell_line, :treatment],
        ["logistic_growth", "gompertz_growth"],
        [:r, :K],                             # share both parameters now
        Dict{Symbol,Float64}(),
        Dict(:K => ("initial_fit", :K))       # inherit carrying capacity from initial fit
    )
]

# Run the staged pipeline
results = run_staged_pipeline(
    experimental_data,
    stages=stages,
    config=config
)
```

## Parameter Uncertainty Quantification

Estimate parameter uncertainty using bootstrap resampling:

```julia
using GrowthParameterEstimation.Workflow

# Fit a model to get baseline parameters
spec = Registry.get_model("logistic_growth")
conditions = build_conditions(experimental_data)

# Perform bootstrap uncertainty estimation
uncertainty = bootstrap_stage_uncertainty(
    spec,
    stage_df;                     # data for this condition/stage
    condition_cols=[:dose, :cell_line],
    shared_params=[:r],           # parameters to estimate uncertainty for
    fixed_params=Dict{Symbol,Float64}(),
    tie_constraints=Dict{Symbol,Symbol}(),
    n_bootstrap=100,              # number of bootstrap samples
    n_starts=10,
    maxiters=300,
    weighted=true,
    reltol=1e-8,
    abstol=1e-8,
    seed=2026
)

# Examine uncertainty results
for (param, stats) in uncertainty
    println("$param: mean=$(stats["mean"]), std=$(stats["std"])")
    println("  95% CI: [$(stats["ci_lower"]), $(stats["ci_upper"])]")
end
```

## Model Population Templates

Use predefined templates for common experimental designs:

```julia
using GrowthParameterEstimation.Workflow

# Create stages for naive and resistant cell populations
population_stages = default_population_stages(["naive", "resistant"])

# Create stages with cell-line specificity
# (would need actual data frame here)
# cellline_stages = default_population_cellline_stages(df, 
#                                                     populations=["naive", "resistant"])
```

## Custom Model Registration

Register custom models with the model registry:

```julia
using GrowthParameterEstimation.Registry

# Define a custom ODE model
function my_custom_model!(du, u, p, t, exposure)
    # du[1] = ... your model equations here
    # p[1], p[2] etc. are your parameters
    # exposure is the exposure function affecting dynamics
end

# Create a model specification
my_spec = ModelSpec(
    name="my_custom_model",
    ode! = my_custom_model!,
    observation = (state, p, t) -> state[1],  # observe first state
    default_solver = Tsit5(),
    param_names = ["param1", "param2", "param3"],
    bounds = [(0.0, 5.0), (0.0, 10.0), (0.1, 2.0)],
    p0_factory = (r0, K0, dose) -> [0.5, 50.0, 1.0],  # function to generate initial guesses
    n_states = 1
)

# Register the model
register_model!(my_spec)

# Now you can use it like any other model
result = fit_model(
    Registry.get_model("my_custom_model"),
    time_data,
    observation_data,
    dose=2.5
)
```

## Joint Fitting Across Multiple Datasets

Fit a single parameter vector to multiple datasets simultaneously:

```julia
using GrowthParameterEstimation.Fitting

# Define datasets (could be different observables or conditions)
dataset_specs = [
    (x = time_points1, y = obs_data1, state_index = 1),  # first state observable
    (x = time_points2, y = obs_data2, state_index = 2)   # second state observable
]

# Initial conditions and parameter guess
u0 = [1.0, 0.0]  # [state1, state2]
p0 = [0.5, 0.1, 20.0]  # [rate1, rate2, coupling]

# Define a coupled ODE model
function coupled_model!(du, u, p, t)
    du[1] = p[1] * u[1] - p[3] * u[1] * u[2]  # state1 dynamics
    du[2] = p[2] * u[2] + p[3] * u[1] * u[2]  # state2 dynamics
end

# Perform joint fitting
joint_result = run_joint_fit(
    coupled_model!,
    dataset_specs,
    u0,
    p0;
    solver=Tsit5(),
    maxiters=5000
)

println("Joint fitting results:")
println("Parameters: $(joint_result.params)")
println("BIC: $(joint_result.bic)")
```

## Resuming Long-Running Workflows

Save and resume long-running pipeline executions:

```julia
using GrowthParameterEstimation.Workflow

# Run pipeline and save manifest
results = run_staged_pipeline(data, stages=stages, config=config)
manifest_path = save_run_manifest("pipeline_manifest.toml"; 
                               config=config,
                               stage_results=results.stages,
                               parameter_bank=results.parameter_bank)

# Later, resume from where you left off
# (assuming you want to resume after stage 2)
resume_results = run_staged_pipeline(
    data,
    stages=stages,
    config=config,
    resume_from_stage="stage_3_name",  # or use resume_manifest_path
    resume_manifest_path="pipeline_manifest.toml"
)
```