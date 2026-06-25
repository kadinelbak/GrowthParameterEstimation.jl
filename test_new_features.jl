using GrowthParameterEstimation

# Test building a simple model
model = build_logistic(r=0.5, K=1e9)
println("Built logistic model: ", model)
println("Model prediction at u=1e6: ", model(1e6, nothing, 0.0))

# Test applying modifiers
model_with_death = apply_death(model; death_rate=0.1)
println("Model with death: ", model_with_death)
println("Model with death prediction: ", model_with_death(1e6, nothing, 0.0))

# Test composing multiple modifiers
final_model = compose_models(
    build_logistic(r=0.5, K=1e9),
    [DeathModifier, LagPhaseModifier];
    death_rate=0.1, tlag=2.0
)
println("Composed model: ", final_model)
println("Composed model prediction: ", final_model(1e6, nothing, 0.0))

# Test converting to ODE function
ode_func = to_ode!(final_model)
println("ODE function type: ", typeof(ode_func))

# Test registration
spec = register_composable_model("test_model", final_model;
                                bounds=[(1e-6, 2.0), (1e6, 1e10), (0.0, 1.0), (0.0, 5.0)],
                                family="test")
println("Registered model: ", spec.name)
println("Parameter names: ", spec.param_names)
println("Number of states: ", spec.n_states)