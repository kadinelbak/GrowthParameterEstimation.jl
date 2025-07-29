module V1SimpleODE

# Include submodules
include("models.jl")
include("fitting.jl") 
include("analysis.jl")

# Re-export all public functions from submodules
using .Models
using .Fitting
using .Analysis

# Export all public functions
export 
    # Model functions
    logistic_growth!, logistic_growth_with_death!, gompertz_growth!, 
    gompertz_growth_with_death!, exponential_growth_with_delay!, 
    logistic_growth_with_delay!,
    
    # Fitting functions
    setUpProblem, calculate_bic, pQuickStat, run_single_fit, 
    compare_models, compare_datasets, compare_models_dict, fit_three_datasets,
    
    # Analysis functions
    leave_one_out_validation, k_fold_cross_validation, parameter_sensitivity_analysis,
    residual_analysis, enhanced_bic_analysis

end # module V1SimpleODE
