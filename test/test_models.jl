using GrowthParameterEstimation.Models
using GrowthParameterEstimation.Registry

# Test building a simple model
model = build_logistic(r=0.5, K=1e9)
println("Built logistic model: ", model)
println("Model prediction at u=1e6 with p=(0.0,): ", model(1e6, (0.0,), 0.0))  # dummy param

# Test applying modifiers
model_with_death = apply_death(model; death_rate=0.1)
println("Model with death: ", model_with_death)
println("Model with death prediction (p=(0.1,)): ", model_with_death(1e6, (0.1,), 0.0))
println("Model with death prediction (p=(0.2,)): ", model_with_death(1e6, (0.2,), 0.0))

# Test applying lag modifier
model_with_lag = apply_lag(model; tlag=2.0)
println("Model with lag: ", model_with_lag)
println("Model with lag prediction (t=1, p=(0.0,)): ", model_with_lag(1e6, (0.0,), 1.0))
println("Model with lag prediction (t=3, p=(0.0,)): ", model_with_lag(1e6, (0.0,), 3.0))

# Test applying hill inhibition
model_with_inhib = apply_hill_inhibition(model; emax=0.8, ic50=0.5, hill=2.0)
println("Model with inhibition: ", model_with_inhib)
println("Model with inhibition prediction (p=(0.8,0.5,2.0), drug=0.0): ", model_with_inhib(1e6, (0.8,0.5,2.0), 0.0))
println("Model with inhibition prediction (p=(0.8,0.5,2.0), drug=1.0): ", model_with_inhib(1e6, (0.8,0.5,2.0), 1.0))

# Test composing multiple modifiers
final_model = compose_models(
    build_logistic(r=0.5, K=1e9),
    [DeathModifier, LagPhaseModifier];
    death_rate=0.1, tlag=2.0
)
println("Composed model: ", final_model)
println("Composed model prediction (t=1, p=(0.1,)): ", final_model(1e6, (0.1,), 1.0))
println("Composed model prediction (t=3, p=(0.1,)): ", final_model(1e6, (0.1,), 3.0))

# Test converting to ODE function
ode_func = to_ode!(final_model)
println("ODE function type: ", typeof(ode_func))
# Test the ODE function with a dummy exposure parameter (which it ignores)
du = [0.0]
u = [1e6]
p = (0.1,)  # death_rate
t = 3.0
ode_func(du, u, p, t)
println("ODE function result du[1]: ", du[1])

# Test registration
spec = register_composable_model("test_model", final_model;
                                bounds=[(1e-6, 2.0), (1e6, 1e10), (0.0, 1.0), (0.0, 5.0)],
                                family="test")
println("Registered model: ", spec.name)
println("Parameter names: ", spec.param_names)
println("Number of states: ", spec.n_states)
