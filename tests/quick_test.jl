using GrowthParameterEstimation

# Simple test
x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [5.0, 8.0, 12.0, 18.0, 22.0]

println("Testing single fit with bounds...")
result = run_single_fit(x, y, [0.5, 25.0]; bounds=[(0.01, 2.0), (10.0, 50.0)], show_stats=true)

println("Success! Result: ", result.params)
println("BIC: ", result.bic)
