# Global, Paired, and Hierarchical Identifiability

## Objective

Extend the practical-identifiability suite for complex multi-condition ODE
models without replacing the existing bootstrap confidence-interval workflow.
The new analyses must work with the same `dataset_specs` representation used
by `run_joint_fit`.

## Public APIs

### Global sensitivity

`global_sensitivity_analysis(model, dataset_specs, u0; bounds, parameter_names, ...)`

- Draws reproducible Latin-hypercube parameter samples inside explicit bounds.
- Supports log-scale sampling for strictly positive parameters.
- Simulates every measured series and returns pointwise partial rank
  correlations (PRCCs), parameter summaries, and failed simulations.
- Is an exploration of the stated parameter ranges, not a posterior analysis.

### Paired profile likelihood

`paired_profile_likelihood(model, dataset_specs, u0, p0; bounds, parameter_names, pair, ...)`

- Fixes two selected parameters on a two-dimensional grid.
- Re-fits all remaining parameters at each grid point.
- Returns the weighted-SSE surface and the two-parameter 95% acceptance region
  using a default chi-square threshold of 5.991.
- Reports whether the accepted region reaches either supplied bound.

### Hierarchical joint fit

`hierarchical_joint_fit(model, groups, p0; bounds, parameter_names, varying_parameters, ...)`

- Fits all named groups in one objective.
- Estimates a central parameter vector and mean-centered, partially pooled
  group deviations for selected varying parameters.
- Uses a Gaussian penalty on log-scale deviations for positive parameters.
- Returns group-specific parameter vectors, per-group data SSE, predictions,
  pooled BIC, and the penalized objective.

## Data contract

Each group is a named tuple with:

```julia
(name = "sensitive", dataset_specs = datasets, u0 = initial_state)
```

Each dataset follows the established contract:

```julia
(x = time, y = observations, state_index = 1, residual_scale = measurement_sd)
```

`residual_scale` is required conceptually for credible cross-assay weighting;
the default remains `1.0` for compatibility.

## Acceptance criteria

- All three APIs are exported from the package and documented.
- Global sensitivity produces a complete sample audit and a parameter summary.
- Paired profiling re-fits nuisance parameters and flags bound-touching regions.
- Hierarchical fitting uses all groups together and exposes both central and
  group-specific parameters.
- Focused tests cover deterministic logistic examples, output shapes, and
  basic failure checks.
- The prior residual-bootstrap 95% interval API remains unchanged.
