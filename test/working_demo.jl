using V1SimpleODE
using Plots

println("🚀 V1SimpleODE Package Demonstration")
println("="^50)

# Generate realistic synthetic logistic growth data
println("📊 Generating realistic logistic growth data...")
true_r, true_K = 0.3, 100.0
x_data = collect(0.0:1.0:10.0)  # Time points
y_true = [true_K / (1 + ((true_K - 5.0) / 5.0) * exp(-true_r * t)) for t in x_data]
y_data = y_true .+ randn(length(y_true)) .* 2.0  # Add noise
y_data = max.(y_data, 1.0)  # Ensure positive values

println("✓ Data generated:")
println("  Time points: ", x_data)
println("  Measurements: ", round.(y_data, digits=2))
println("  True parameters: r=$(true_r), K=$(true_K)")

# 1. Basic single fit with bounds for stability
println("\n🔧 1. Testing basic single fit...")
p0 = [0.1, 80.0]  # Initial parameter guess
bounds = [(0.01, 2.0), (10.0, 150.0)]  # Reasonable bounds

result = run_single_fit(x_data, y_data, p0; bounds=bounds, show_stats=true)

println("✓ Single fit completed!")
println("  Fitted parameters: r=$(round(result.params[1], digits=4)), K=$(round(result.params[2], digits=2))")
println("  BIC: $(round(result.bic, digits=2))")
println("  SSR: $(round(result.ssr, digits=4))")

# 2. Model comparison demonstration
println("\n🔧 2. Testing model comparison...")
compare_models(
    x_data, y_data,
    "Logistic", logistic_growth!, [0.1, 80.0],
    "Gompertz", gompertz_growth!, [0.1, 1.0, 80.0];
    bounds1=[(0.01, 2.0), (10.0, 150.0)],
    bounds2=[(0.01, 2.0), (0.1, 10.0), (10.0, 150.0)],
    show_stats=true
)

println("✓ Model comparison completed!")

# 3. Three datasets demonstration with proper data
println("\n🔧 3. Testing three datasets fitting...")

# Generate three different realistic datasets
x1 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
x2 = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
x3 = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

# Dataset 1: Fast growth
y1_true = [50.0 / (1 + ((50.0 - 3.0) / 3.0) * exp(-0.4 * t)) for t in x1]
y1 = y1_true .+ randn(length(y1_true)) .* 1.0
y1 = max.(y1, 1.0)

# Dataset 2: Slow growth
y2_true = [80.0 / (1 + ((80.0 - 5.0) / 5.0) * exp(-0.2 * t)) for t in x2]
y2 = y2_true .+ randn(length(y2_true)) .* 1.5
y2 = max.(y2, 1.0)

# Dataset 3: Medium growth
y3_true = [60.0 / (1 + ((60.0 - 4.0) / 4.0) * exp(-0.25 * t)) for t in x3]
y3 = y3_true .+ randn(length(y3_true)) .* 1.2
y3 = max.(y3, 1.0)

println("  Dataset 1: ", length(x1), " points, max value: ", round(maximum(y1), digits=2))
println("  Dataset 2: ", length(x2), " points, max value: ", round(maximum(y2), digits=2))
println("  Dataset 3: ", length(x3), " points, max value: ", round(maximum(y3), digits=2))

three_results = fit_three_datasets(
    x1, y1, "Fast Growth",
    x2, y2, "Slow Growth", 
    x3, y3, "Medium Growth",
    [0.1, 70.0];
    bounds=[(0.01, 2.0), (10.0, 100.0)],
    show_stats=true
)

println("✓ Three datasets fitting completed!")
println("  Dataset 1 fit: r=$(round(three_results.fit1.params[1], digits=3)), K=$(round(three_results.fit1.params[2], digits=1))")
println("  Dataset 2 fit: r=$(round(three_results.fit2.params[1], digits=3)), K=$(round(three_results.fit2.params[2], digits=1))")
println("  Dataset 3 fit: r=$(round(three_results.fit3.params[1], digits=3)), K=$(round(three_results.fit3.params[2], digits=1))")

# 4. Cross-validation example
println("\n🔧 4. Testing cross-validation...")
cv_result = leave_one_out_validation(x_data, y_data, [0.1, 80.0]; 
                                   bounds=bounds, show_stats=false)
println("✓ Cross-validation completed!")
println("  RMSE: $(round(cv_result.rmse, digits=4))")
println("  R²: $(round(cv_result.r_squared, digits=3))")
println("  Valid predictions: $(cv_result.n_valid)/$(length(x_data))")

# 5. Basic analysis demonstrations
println("\n🔧 5. Testing analysis functions...")

# Enhanced BIC analysis
println("Running enhanced BIC analysis...")
try
    bic_result = enhanced_bic_analysis(x_data, y_data; show_plots=false)
    println("✓ Enhanced BIC analysis completed!")
    println("  Number of successful fits: $(length(bic_result.results))")
    if !isempty(bic_result.results)
        best_idx = findmin([res.bic for res in bic_result.results])[2]
        best_model = bic_result.results[best_idx]
        println("  Best model: $(best_model.name)")
        println("  Best model BIC: $(round(best_model.bic, digits=2))")
    end
catch e
    println("⚠ Enhanced BIC analysis failed: $(e)")
end
    println("  Best BIC: $(round(best_model.bic, digits=2))")
end

println("\n🎉 Demonstration completed successfully!")
println("Check the generated plots and CSV files for detailed results.")
println("="^50)
