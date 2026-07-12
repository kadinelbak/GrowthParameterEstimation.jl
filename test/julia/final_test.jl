using GrowthParameterEstimation.Models

# Test basic functionality
println("Testing basic model building...")
mod = build_logistic(r=0.5, K=1e9)
println("Created logistic model: ", mod)

println("\nTesting model evaluation...")
result = mod(1e6, (0.0,), 0.0)  # u=1e6, no params, t=0
println("Model prediction at u=1e6: ", result)

println("\nTesting death modifier...")
mod_death = apply_death(mod; death_rate=0.1)
println("Created model with death: ", mod_death)
result_death = mod_death(1e6, (0.1,), 0.0)  # u=1e6, death_rate=0.1, t=0
println("Model prediction with death: ", result_death)

println("\nTesting lag modifier...")
mod_lag = apply_lag(mod; tlag=2.0)
println("Created model with lag: ", mod_lag)
result_lag_t1 = mod_lag(1e6, (0.0,), 1.0)   # t=1.0 < tlag, should be 0
result_lag_t3 = mod_lag(1e6, (0.0,), 3.0)   # t=3.0 > tlag, should be normal
println("Model prediction at t=1.0 (before lag): ", result_lag_t1)
println("Model prediction at t=3.0 (after lag): ", result_lag_t3)

println("\nAll tests completed successfully!
