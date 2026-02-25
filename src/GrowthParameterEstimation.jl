module GrowthParameterEstimation

# Include submodules
include("data.jl")
include("exposure.jl")
include("models.jl")
include("registry.jl")
include("simulation.jl")
include("observation.jl")
include("fitting.jl")
include("analysis.jl")
include("workflow.jl")

using .DataLayer
using .Exposure
using .Models
using .Registry
using .Simulation
using .Observation
using .Fitting
using .Analysis
using .Workflow

export
    # Data layer
    REQUIRED_COLUMNS, load_timeseries, normalize_schema, validate_timeseries,

    # Drug exposure layer
    AbstractExposure, ConstantExposure, PulseExposure, SteppedExposure, DecayingExposure,
    build_exposure, evaluate_exposure,

    # Model registry + simulation API
    ModelSpec, register_model, get_model, list_models,
    SimulationResult, simulate,

    # Observation API
    ObservationSpec, observed_signal, viable_total, sum_states,

    # Model functions
    logistic_growth!, logistic_growth_with_death!, gompertz_growth!,
    gompertz_growth_with_death!, exponential_growth_with_delay!,
    logistic_growth_with_delay!, exponential_growth!,
    exponential_growth_with_death_and_delay!,

    # Fitting functions
    setUpProblem, calculate_bic, pQuickStat, run_single_fit,
    compare_models, compare_datasets, compare_models_dict, fit_three_datasets,

    # Analysis functions
    leave_one_out_validation, k_fold_cross_validation, parameter_sensitivity_analysis,
    residual_analysis, enhanced_bic_analysis,

    # End-to-end workflow APIs
    FitCondition, PipelineConfig,
    default_config, save_config, load_config,
    build_conditions, fit, rank_models, plot_topk, export_results, run_pipeline

function __init__()
    Registry.register_builtin_models!()
end

end # module GrowthParameterEstimation
