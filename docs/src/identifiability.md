# Identifiability analysis

Parameter estimation and identifiability are different questions. A low
residual error only shows that one parameter vector fits the observed data; it
does not establish that the parameters are uniquely supported by the data.

`GrowthParameterEstimation.jl` provides practical, numerical diagnostics for
arbitrary `run_joint_fit` models. Structural global/local identifiability is
kept separate because it needs a symbolic differential-algebra analysis of the
ODEs and the exact observation process.

## 1. State the observation map

Document every state and every quantity that the assay actually measures.
For a four-population model, one useful starting point is
`S` (viable sensitive cells), `D1` (recoverably damaged cells), `D2` (terminal
damage/dead cells), and `M` (macrophages). Directly measuring `S`, `D1`, `D2`,
and `M` gives a substantially different identifiability result from measuring
only total viability.

```julia
# This is an illustrative model. Drug concentration is an externally known
# input, while the seven entries of p are estimated from the joint data.
drug_concentration(t) = t <= 2.0 ? 1.0 : 0.2

function four_state_model!(du, u, p, t)
    S, D1, D2, M = u
    growth, capacity, drug_damage, repair, terminal_damage, macrophage_clearance, macrophage_killing = p
    damage = drug_damage * drug_concentration(t) * S

    du[1] = growth * S * (1 - (S + D1) / capacity) - damage - macrophage_killing * M * S
    du[2] = damage - repair * D1 - terminal_damage * D1
    du[3] = terminal_damage * D1 - macrophage_clearance * M * D2
    du[4] = 0.0  # Replace with macrophage recruitment/dynamics when those terms are supported by data.
end

map = ObservationMap(
    "drug_macrophage_four_state",
    [:S, :D1, :D2, :M],
    [:viable_cells, :recoverable_damage, :terminal_damage, :macrophages];
    description = "Each assay maps directly to one modeled state.",
)

datasets = [
    (x = time, y = S_counts, state_index = 1, residual_scale = S_sd),
    (x = time, y = D1_counts, state_index = 2, residual_scale = D1_sd),
    (x = time, y = D2_counts, state_index = 3, residual_scale = D2_sd),
    (x = time, y = M_counts, state_index = 4, residual_scale = M_sd),
]

initial_state = [S_counts[1], D1_counts[1], D2_counts[1], M_counts[1]]

validate_observation_map(map, datasets)
```

Use one `dataset_specs` entry for each independently measured series. Set
`residual_scale` to a defensible measurement SD (or a scale derived from
replicates). It supplies the weighting used by `run_joint_fit`, Fisher
information, profiles, and simulated recovery tests.

## 2. Run the practical-identifiability report

```julia
config = IdentifiabilityConfig(
    [:growth, :capacity, :drug_damage, :repair, :terminal_damage,
     :macrophage_clearance, :macrophage_killing],
    [(0.01, 2.0), (1e3, 1e7), (1e-5, 5.0),
     (1e-5, 5.0), (1e-5, 5.0), (1e-5, 5.0), (1e-7, 5.0)];
    n_starts = 40,
    start_scale = :log,
    profile_points_per_side = 12,
    bootstrap_replicates = 200,
    bootstrap_method = :parametric,
)

initial_guess = [0.3, 1e5, 0.1, 0.1, 0.05, 1e-4, 1e-7]

report = practical_identifiability_report(
    four_state_model!, datasets, initial_state, initial_guess;
    config = config,
    optimizer = :nelder_mead,
    maxiters = 20_000,
)
```

The result contains a best joint fit, multistart audit table, clusters of
near-equivalent solutions, Fisher-information result, and profile likelihoods.
When `bootstrap_replicates > 0`, it also includes a series-stratified bootstrap
summary in `report.bootstrap`; preserve biological/assay replicate strata as
separate dataset series when building that analysis.
Treat `report.status == "passes_numerical_gates"` as a numerical screening
result, not a proof that the biology is identifiable.

## 3. Interpret the diagnostics

- `fisher.numerical_rank < number_of_parameters`: at least one local parameter
  direction is not estimable at the fitted solution.
- Very large `fisher.condition_number`: parameters are nearly confounded even
  when the numerical rank is full.
- More than one solution cluster within the BIC tolerance: different parameter
  vectors fit about equally well.
- A profile confidence status other than `"bounded"`: the selected parameter
  has not been bracketed by the current data and parameter bounds.
- Low bootstrap success rate or a broad/multimodal bootstrap distribution:
  the fitted parameter is not stable to plausible sampling variation.

For difficult systems, report these diagnostics alongside each biological
conclusion rather than quoting a single best-fit parameter vector.

## 4. Test recovery before real-data interpretation

```julia
starts = generate_multistarts(config.bounds; n_starts = 40, scale = :log)
recovery = synthetic_recovery_benchmark(
    four_state_model!, datasets, initial_state, known_parameters;
    bounds = config.bounds,
    starts = starts,
    n_simulations = 200,
)
```

This generates data at the real measurement schedule, adds Gaussian error
using each series' `residual_scale`, and refits every simulated experiment.
Use it to compare alternative time grids, replicate counts, macrophage
perturbations, and direct damaged-cell measurements before committing to an
experimental design.

`bootstrap_joint_fit` provides residual and parametric bootstrap refits for
an already fitted experiment. It resamples within each supplied series, so
replicate-aware analyses should keep biological/assay replicate strata
separate in `dataset_specs`.

## 5. Structural analysis is intentionally explicit

```julia
using StructuralIdentifiability

# Repeat the model in symbolic form. Its output equations must match the
# ObservationMap and datasets used for numerical fitting.
symbolic_model = @ODEmodel(
    S'(t) = growth * S(t) * (1 - (S(t) + D1(t)) / capacity) -
            drug_damage * drug(t) * S(t) - macrophage_killing * M(t) * S(t),
    D1'(t) = drug_damage * drug(t) * S(t) - repair * D1(t) - terminal_damage * D1(t),
    D2'(t) = terminal_damage * D1(t) - macrophage_clearance * M(t) * D2(t),
    M'(t) = 0,
    yS(t) = S(t),
    yD1(t) = D1(t),
    yD2(t) = D2(t),
    yM(t) = M(t),
)

structural = structural_identifiability(
    symbolic_model;
    mode = :global,
    prob_threshold = 0.99,
)
```

The table labels each term as `"globally"`, `"locally"`, or
`"nonidentifiable"`. For a faster screening run, use `mode = :local`; use
`experiment_type = :ME` only when analyzing independent initial conditions
across multiple experiments. The symbolic model must include known inputs such
as `drug(t)` and the same outputs, initial-condition assumptions, and
observation map used for fitting. A numerical optimizer, FIM, bootstrap, or
profile cannot prove structural identifiability by itself.
