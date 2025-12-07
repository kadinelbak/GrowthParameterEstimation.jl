module GrowthParameterEstimation

# Include submodules
include("models.jl")
include("fitting.jl")
include("analysis.jl")

using .Models
using .Fitting
using .Analysis

export
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
    residual_analysis, enhanced_bic_analysis

end # module GrowthParameterEstimation
