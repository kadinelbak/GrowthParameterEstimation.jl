module GUICustomModels

using GrowthParameterEstimation

function logistic_with_death_ode!(du, u, p, t, exposure)
    N = max(u[1], 0.0)
    r, K, d = p
    dose = max(exposure(t), 0.0)
    du[1] = r*N*(1 - N/K) - d*N
    return nothing
end

function register_custom_models!(; overwrite::Bool = true)
    register_model!(ModelSpec(
        name = "Logistic with Death",
        ode! = logistic_with_death_ode!,
        param_names = [:r, :K, :d],
        bounds = [(1.0e-6, 5.0), (0.001, 1.0e7), (-2.0, 1.0)],
        n_states = 1,
        observable = u -> u[1],
        base_growth_family = "custom_logistic",
    ); overwrite = overwrite)
    return nothing
end

end # module
