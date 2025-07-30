# Simple test runner for development testing
# This file can be run directly with: julia simple_test.jl

push!(LOAD_PATH, joinpath(@__DIR__, ".."))

try
    using GrowthParamEst
    println("✓ GrowthParamEst package loaded successfully")
    
    # Test basic functionality with simple data
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    y = [5.0, 8.0, 15.0, 28.0, 45.0, 65.0]
    
    println("\nTesting basic fitting...")
    result = run_single_fit(x, y, [0.1, 70.0]; show_plots=false, show_stats=false)
    println("✓ Basic fitting works - SSR: $(round(result.ssr, digits=4))")
    
    println("\nTesting model comparison...")
    comparison = compare_models(x, y; show_plots=false)
    println("✓ Model comparison works - Best model: $(comparison.best_model.name)")
    
    println("\nTesting analysis functions...")
    # Test leave-one-out validation
    loo_result = leave_one_out_validation(x, y, [0.1, 70.0]; show_stats=false)
    println("✓ Leave-one-out validation works - RMSE: $(round(loo_result.rmse, digits=4))")
    
    # Test sensitivity analysis
    sens_result = parameter_sensitivity_analysis(x, y, result; show_plots=false)
    println("✓ Sensitivity analysis works - $(length(sens_result.sensitivity_metrics)) parameters analyzed")
    
    # Test residual analysis
    resid_result = residual_analysis(x, y, result; show_plots=false)
    println("✓ Residual analysis works - $(resid_result.statistics.n_outliers) outliers found")
    
    # Test BIC analysis
    bic_result = enhanced_bic_analysis(x, y; show_plots=false)
    println("✓ BIC analysis works - $(length(bic_result.successful_results)) successful model fits")
    
    println("\n" * "="^50)
    println("All basic tests passed! ✓")
    println("The GrowthParamEst package is working correctly.")
    println("="^50)
    
catch e
    println("✗ Error during testing: $e")
    println("\nStacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end
