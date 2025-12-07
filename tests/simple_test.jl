using GrowthParameterEstimation

# Test with very simple synthetic data that we know should work
x_test = [0.0, 1.0, 2.0, 3.0, 4.0]
y_test = [5.0, 8.0, 12.0, 18.0, 22.0]  # Simple logistic-like growth

println("Testing with simple data:")
println("x: ", x_test)
println("y: ", y_test)

# Test with better bounds
result = run_single_fit(
    x_test, y_test, [0.5, 25.0]; 
    bounds = [(0.001, 2.0), (10.0, 50.0)],
    show_stats = true
)

println("Result params: ", result.params)
println("BIC: ", result.bic)
