# Research use

`GrowthParameterEstimation.jl` was written for, and is used in, mathematical
oncology research on drug resistance in ovarian cancer cell lines. Every claim on
this page points to a file you can check. Two of the research repositories are
private for now (unpublished data), and this page says so where it applies.

## Projects that use the package

| Project | Biological question | Data | Package version used | Code |
| --- | --- | --- | --- | --- |
| A2780 cisplatin resistance and dose scheduling | How cisplatin-sensitive (A2780) and cisplatin-resistant (A2780cis) ovarian cancer cells grow, compete and respond to cisplatin, and which dosing schedule best delays progression | Time-lapse cell counts, monoculture and coculture, untreated and cisplatin-treated | `v0.4.x`, pinned to commit `5eaadd4` | `kadinelbak/CancerGrowthDynamics` (private); results site [kadinelbak.github.io/CancerGrowthDynamics](https://kadinelbak.github.io/CancerGrowthDynamics/) |
| OVCAR3 olaparib resistance and glutamine | Whether OVCAR3 ovarian cancer cells move from an olaparib-sensitive to a resistant state, and whether glutamine starvation changes that move | 48-well adaptive-therapy plate (September 2026): sensitive and resistant lines, vehicle / constant 20 µM / adaptive olaparib, each with and without glutamine | `v0.5.4`, pinned to commit `c6fa55d` | `kadinelbak/parpResearch` (private) |

The package has been a dependency of the A2780 project since February 2026
(first commit adding it: `264f459`, 2026-02-08, "Monoculture simple").

## What each project uses

### A2780 pipeline (`CancerGrowthDynamics/Modeling_Approaches/02_mechanical_automatic_package`)

A staged pipeline fits a family of autonomous compartment models (no explicit
time dependence, so each model can be driven by any dose schedule) and passes
the selected model to an optimal-control step for the cisplatin schedule.

| Package function | Where it is used | Purpose |
| --- | --- | --- |
| `run_joint_fit` | `bootstrap_tools.jl` (`bs_fit`, called by `fit_cand` for every candidate in every stage) | Joint fit of one multi-state model across all trajectories of a stage, with an estimated initial state (`u0_builder`) where the first count is far from the seeding |
| `generate_multistarts`, `run_joint_multistart` | `zoo_heldout.jl` | Multistart refits for held-out validation |
| `evaluate_joint_fit` | `zoo_heldout.jl` | Scoring held-out predictions |

The pipeline also follows the schemes of `bootstrap_joint_fit`,
`parameter_sensitivity_analysis` and `blocked_joint_validation`, but
reimplements them in `zoo_analyses.jl`: `bootstrap_joint_fit` cannot yet
resample weighted data, `parameter_sensitivity_analysis` handles one-state
models only, and the blocked validation is run locally so that refits use the
same residual weights as the main fits (see the comments and method notes in
`zoo_analyses.jl`, lines 8–14, 109, 410, 758 and 871). These are known gaps.

The earlier manual approach (`Modeling_Approaches/01_mechanical_manual`) also
lists the package as a dependency. The neural-network track
(`03_automatic_neural_network`) deliberately does not use it.

### OVCAR3 pipeline (`parpResearch`)

The README of that repository states that "the fitting, multistart, bootstrap,
profile-likelihood, Fisher-information and synthetic-recovery machinery is
GrowthParameterEstimation.jl", and that the repository holds only the
experiment-specific parts (data loading, the population model, stage
configuration and reporting).

| Package function | Where it is used | Purpose |
| --- | --- | --- |
| `generate_multistarts`, `run_joint_multistart` | `src/fitting.jl` | One multistart joint fit per candidate model per stage (stages O1–O5, 16 transition models at stage O4) |
| `run_joint_fit` | `scripts/presentation.jl`, `scripts/extended_window.jl`, `scripts/identifiability_o4.jl` | Locked-parameter and refit variants of the stage models |
| `bootstrap_joint_fit` | `src/analyses.jl` | Residual bootstrap intervals |
| `profile_likelihood` | `src/analyses.jl` | Profile-likelihood confidence intervals for the selected models |
| `fisher_information` | `src/analyses.jl` | Fisher-information intervals and identifiability checks |
| `synthetic_recovery_benchmark` | `src/analyses.jl` | Parameter recovery from data simulated with the fitted model |

## Outputs that rely on the package

- The A2780 results site (public):
  [kadinelbak.github.io/CancerGrowthDynamics](https://kadinelbak.github.io/CancerGrowthDynamics/).
  Its staged model-comparison, bootstrap, sensitivity and validation reports
  name the package as the fitting engine.
- An undergraduate honors thesis (University of Florida, Biomedical
  Engineering), in preparation, whose methods section describes fitting with
  `GrowthParameterEstimation.jl`.
- A manuscript on the A2780 work, in preparation.

## Citing

If you use the package, please cite it with the metadata in
[`CITATION.cff`](https://github.com/kadinelbak/GrowthParameterEstimation.jl/blob/main/CITATION.cff).
If you use it in your own research, we would be glad to list your project here:
open an issue or a pull request.
