# Models module - Contains all ODE model definitions
module Models

export logistic_growth!, logistic_growth_with_death!, gompertz_growth!, 
       gompertz_growth_with_death!, exponential_growth_with_delay!, 
       logistic_growth_with_delay!

# 1) plain logistic: p = (r, K)
function logistic_growth!(du,u,p,t)
  r,K = p; du[1] = r*u[1]*(1 - u[1]/K)
end

# 2) logistic + death: p = (r, K, δ)
function logistic_growth_with_death!(du,u,p,t)
  r,K,δ = p; du[1] = r*u[1]*(1 - u[1]/K) - δ*u[1]
end

# 3) Gompertz: p = (a, b)
function gompertz_growth!(du,u,p,t)
  a,b = p; du[1] = a*u[1]*exp(-b*t)
end

# 4) Gompertz + death: p = (a, b, δ)
function gompertz_growth_with_death!(du,u,p,t)
  a,b,δ = p; du[1] = a*u[1]*exp(-b*t) - δ*u[1]
end

# 5) exp with lag: p = (r, t_lag)
function exponential_growth_with_delay!(du,u,p,t)
  r,tlag = p; du[1] = (t>=tlag ? r : 0.0)*u[1]
end

# 6) logistic with lag: p = (r, K, t_lag)
function logistic_growth_with_delay!(du,u,p,t)
  r,K,tlag = p; du[1] = (t>=tlag ? r : 0.0)*u[1]*(1-u[1]/K)
end

end # module Models
