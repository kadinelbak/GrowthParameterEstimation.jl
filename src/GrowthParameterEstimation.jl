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
    REQUIRED_COLUMNS, STRICT_REQUIRED_METADATA, load_timeseries, normalize_schema, validate_timeseries,

    validate_required_metadata,
    # Drug exposure layer
    AbstractExposure, ConstantExposure, PulseExposure, SteppedExposure, DecayingExposure,
    build_exposure, evaluate_exposure,

    # Model registry + simulation API
    ModelSpec, register_model, get_model, list_models,
    SimulationResult, SweepGrid, SweepResult, simulate, run_sweep,

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
    run_joint_fit, compare_joint_models_dict,

    # Analysis functions
    leave_one_out_validation, k_fold_cross_validation, parameter_sensitivity_analysis,
    residual_analysis, enhanced_bic_analysis,

    # End-to-end workflow APIs
    FitCondition, PipelineConfig, PipelineStage,
    default_config, save_config, load_config,
    default_stages, default_population_stages, default_population_cellline_stages, summarize_datasets,
    validate_strict_schema, generate_qc_report, save_qc_report,
    preflight_data_quality, save_preflight_report,
    save_run_manifest, load_run_manifest, bootstrap_stage_uncertainty,
    build_conditions, fit, rank_models, plot_topk, export_results, run_pipeline, run_staged_pipeline

function __init__()
    Registry.register_builtin_models!()
end

end # module GrowthParameterEstimation
