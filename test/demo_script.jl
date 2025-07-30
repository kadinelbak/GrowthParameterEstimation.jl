# Demonstration script for GrowthParamEst package
# This script shows how to use the package with actual data and plots

using GrowthParamEst
using Plots
using Random

println("🚀 GrowthParamEst Package Demonstration")
println("="^50)

# Set random seed for reproducibility
Random.seed!(42)

# Generate some synthetic logistic growth data
println("\n📊 Generating synthetic test data...")
x_data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
# True logistic with r=0.3, K=100, some noise
true_r, true_K = 0.3, 100.0
y_true = [true_K / (1 + ((true_K - 5.0) / 5.0) * exp(-true_r * t)) for t in x_data]
# Add some realistic noise
noise = randn(length(y_true)) .* 2.0
y_data = y_true .+ noise
y_data = max.(y_data, 1.0)  # Ensure positive values

println("✓ Data generated:")
println("  Time points: ", x_data)
println("  Measurements: ", round.(y_data, digits=2))
println("  True parameters: r=$(true_r), K=$(true_K)")

# 1. Basic single fit demonstration
println("\n🔧 1. Testing basic single fit...")
p0 = [0.1, 80.0]  # Initial parameter guess
result = run_single_fit(x_data, y_data, p0; show_stats=true)

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
    show_stats=true
)

println("✓ Model comparison completed!")

# 3. Three datasets demonstration
println("\n🔧 3. Testing three datasets fitting...")
# Generate three different datasets
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
y3_true = [120.0 / (1 + ((120.0 - 8.0) / 8.0) * exp(-0.25 * t)) for t in x3]
y3 = y3_true .+ randn(length(y3_true)) .* 2.0
y3 = max.(y3, 1.0)

println("  Dataset 1: ", length(x1), " points, max value: ", round(maximum(y1), digits=2))
println("  Dataset 2: ", length(x2), " points, max value: ", round(maximum(y2), digits=2))
println("  Dataset 3: ", length(x3), " points, max value: ", round(maximum(y3), digits=2))

three_results = fit_three_datasets(
    x1, y1, "Fast Growth",
    x2, y2, "Slow Growth", 
    x3, y3, "Medium Growth",
    [0.1, 70.0];
    show_stats=true
)

println("✓ Three datasets fitting completed!")
println("  Dataset 1 fit: r=$(round(three_results.fit1.params[1], digits=3)), K=$(round(three_results.fit1.params[2], digits=1))")
println("  Dataset 2 fit: r=$(round(three_results.fit2.params[1], digits=3)), K=$(round(three_results.fit2.params[2], digits=1))")
println("  Dataset 3 fit: r=$(round(three_results.fit3.params[1], digits=3)), K=$(round(three_results.fit3.params[2], digits=1))")

println("\n🎉 Demonstration completed successfully!")
println("Check the generated plots to see the visual results.")
println("="^50)
