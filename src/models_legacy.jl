# Legacy ODE RHS function definitions for backward compatibility

"""
logistic_growth!(du, u, p, t)

Simple logistic growth: du = r * u * (1 - u/K)
Parameter vector p = [r, K]
"""
function logistic_growth!(du, u, p, t)
    r, K = p[1], p[2]
    du[1] = r * u[1] * (1 - u[1] / K)
    return nothing
end

"""
logistic_growth_with_death!(du, u, p, t)

Logistic growth with death term: du = r*u*(1 - u/K) - d*u
Parameter vector p = [r, K, d]
"""
function logistic_growth_with_death!(du, u, p, t)
    r, K, d = p[1], p[2], p[3]
    du[1] = r * u[1] * (1 - u[1] / K) - d * u[1]
    return nothing
end

"""

gompertz_growth!(du, u, p, t)

Gompertz growth: du = a * u * log(K / u)
Parameter vector p = [a, b_unused, K]  # b retained for API compatibility
"""
function gompertz_growth!(du, u, p, t)
    a, _, K = p[1], p[2], p[3]
    du[1] = u[1] <= 0 || u[1] >= K ? 0.0 : a * u[1] * log(K / u[1])
    return nothing
end

"""

gompertz_growth_with_death!(du, u, p, t)

Gompertz growth with death: du = a*u*log(K/u) - d*u
Parameter vector p = [a, b_unused, K, d]
"""
function gompertz_growth_with_death!(du, u, p, t)
    a, _, K, d = p[1], p[2], p[3], p[4]
    du[1] = u[1] <= 0 || u[1] >= K ? -d * u[1] : a * u[1] * log(K / u[1]) - d * u[1]
    return nothing
end

"""
exponential_growth!(du, u, p, t)

Simple exponential growth: du = r * u
Parameter vector p = [r]
"""
function exponential_growth!(du, u, p, t)
    r = p[1]
    du[1] = r * u[1]
    return nothing
end

"""
exponential_growth_with_delay!(du, u, p, t)

Exponential growth with a lag phase: du = r*u*(1 - u/K) if t >= tlag else 0
Parameter vector p = [r, K, tlag]
"""
function exponential_growth_with_delay!(du, u, p, t)
    r, K, tlag = p[1], p[2], p[3]
    du[1] = (t >= tlag ? r : 0.0) * u[1] * (1 - u[1] / K)
    return nothing
end

"""
logistic_growth_with_delay!(du, u, p, t)

Logistic growth with a lag phase: du = r*u*(1 - u/K) if t >= tlag else 0
Parameter vector p = [r, K, tlag]
"""
function logistic_growth_with_delay!(du, u, p, t)
    r, K, tlag = p[1], p[2], p[3]
    du[1] = (t >= tlag ? r : 0.0) * u[1] * (1 - u[1] / K)
    return nothing
end

"""
exponential_growth_with_death_and_delay!(du, u, p, t)

Exponential growth with death and lag: du = (r if t>=tlag else 0)*u*(1-u/K) - d*u
Parameter vector p = [r, K, d, tlag]
"""
function exponential_growth_with_death_and_delay!(du, u, p, t)
    r, K, d, tlag = p[1], p[2], p[3], p[4]
    du[1] = (t >= tlag ? r : 0.0) * u[1] * (1 - u[1] / K) - d * u[1]
    return nothing
end
