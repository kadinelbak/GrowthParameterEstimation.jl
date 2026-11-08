# Handing fitted models to control tools

GrowthParameterEstimation identifies and qualifies growth models; it does not
solve optimal-control problems. Mature Julia packages already do that
(for example ModelPredictiveControl.jl for nonlinear MPC, or JuMP / InfiniteOpt.jl
for direct optimal control). The control API is the handoff between the two:
it turns a fitted model into the standard `f(x, u, p)` form and carries its
bootstrap parameter sets so a schedule can be checked for robustness.

## The input replaces the exposure

Registered models take an exposure function, `ode!(du, u, p, t, exposure)`.
A `ControlModel` replaces that exposure with a control input `u`:
`exposure(t)` returns `u[1]` for a single-input model and the input vector `u`
for a model with several inputs. Nothing about the fitted model changes.

## Building a ControlModel

```julia
using GrowthParameterEstimation

spec = get_model("logistic_linear_kill")
fit = fit_model(spec, t, y, 1.0)

cm = controlled_model(spec, fit.params; input_names = [:dose])
```

Any right-hand side in the package convention works, including models fitted
with `run_joint_fit`. Attach bootstrap parameter sets directly from
`bootstrap_joint_fit`:

```julia
boot = bootstrap_joint_fit(model!, datasets, u0, p0; bounds = bounds, n_bootstrap = 200)

cm = controlled_model(model!, boot.fit.params;
    n_states = 2, n_inputs = 2,
    state_names = [:S, :R], input_names = [:dose, :fasting],
    param_samples = boot,
    observable = (x, p) -> [x[1] + x[2], x[2] / (x[1] + x[2])],
    output_names = [:total, :resistant_fraction])
```

Declare `autonomous = false` if the right-hand side uses `t` directly. Schedules
built on such models assume the input started at `t = 0`.

## Functions for control solvers

| Function | Returns | Use |
| --- | --- | --- |
| `control_rhs(cm)` | `(x, u, p = cm.params, t = 0) -> dx` | Continuous dynamics, ForwardDiff-compatible |
| `control_rhs!(cm)` | `(dx, x, u, p, t) -> nothing` | In-place continuous dynamics |
| `mpc_functions(cm)` | `(f!, h!, nx, nu, ny, p)` | `(ẋ, x, u, d, p)` / `(y, x, d, p)` convention of nonlinear MPC packages |
| `discrete_dynamics(cm; dt)` | `(x, u, p = cm.params, t0 = 0) -> x_next` | Zero-order-hold step map (fixed RK4), for shooting and MPC |

```julia
F = discrete_dynamics(cm; dt = 2.0)           # dose changes every 2 days
x_next = F(x, [1.0])

using ForwardDiff
A = ForwardDiff.jacobian(x -> F(x, u), x)     # ∂F/∂x
B = ForwardDiff.jacobian(u -> F(x, u), u)     # ∂F/∂u
```

With ModelPredictiveControl.jl:

```julia
using ModelPredictiveControl
m = mpc_functions(cm)
model = NonLinModel(m.f!, m.h!, 2.0, m.nu, m.nx, m.ny; p = m.p)
```

## Checking a schedule

`InputSchedule` holds a piecewise-constant schedule, the form a control
solution takes when doses change only at decision times. `simulate_schedule`
replays it with the full ODE solver (stopping at every dose change), and
`schedule_ensemble` replays it on the point estimate and every bootstrap set.

```julia
sched = InputSchedule(0:2:12, [0, 0, 0, 0.62, 1.47, 1.47, 0])
out = simulate_schedule(cm, x0, sched, 0:0.5:14)

runs = schedule_ensemble(cm, x0, sched, 0:0.5:14)
finals = [r.outputs[1, end] for r in runs]    # spread of the final burden
```

## Exporting

`control_model_summary(cm)` returns names, dimensions, estimates, samples and
metadata as plain data; `save_control_model(cm, path)` writes it as JSON for
tools in other languages.

## API

```@docs
ControlModel
controlled_model
control_rhs
control_rhs!
mpc_functions
discrete_dynamics
InputSchedule
simulate_schedule
schedule_ensemble
parameter_sets
control_model_summary
save_control_model
```
