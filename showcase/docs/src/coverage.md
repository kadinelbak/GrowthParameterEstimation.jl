# What the showcase covers

The showcase fits every curve separately with one recipe. That keeps the datasets comparable, but it means most of the package is not exercised here yet. This page lists what is shown and what is not, so the gaps are visible.

## Shown on the dataset pages

| Package feature | Where |
| --- | --- |
| Model registry (`get_model`): 18 of the 34 built-in models, all single-population | every page |
| Multistart fitting of one curve (`fit`, `optimizer = :transformed`) | every page |
| AIC, AICc and BIC from each fit, with the 2-BIC selection rule | every page |
| Profile likelihood with the other parameters refitted (`fit(...; fixed_params)`) | every page except BT-474 |
| Simulation and forecasting (`simulate`) | every page |
| Time-varying exposure: on/off therapy (`SteppedExposure`) | Bruchovsky PSA, Ebos sunitinib |
| Dose-step schedules per group | Ebos sunitinib |
| Treatment-response laws (kill terms, Stein, Claret) | Ghaffari Laleh, Bruchovsky, Ebos |
| Fisher information and Wald intervals (`fisher_information`) | "Package diagnostics" on every page |
| Built-in profile likelihood (`profile_likelihood`), with its default grid and with a Fisher-scaled grid | "Package diagnostics" on every page |
| Residual bootstrap (`bootstrap_joint_fit`) | "Package diagnostics" on every page |
| Synthetic recovery on each curve's own design (`synthetic_recovery_benchmark`) | "Package diagnostics" on every page |
| Cross-validation (`k_fold_cross_validation`) | "Package diagnostics" on every page |

## Not shown yet

| Package feature | What it would add | Natural dataset |
| --- | --- | --- |
| Joint fits with shared parameters (`fit(...; shared_params, tie_constraints)`, `run_joint_fit`) | one drug effect estimated from many weak curves instead of one per curve | Ebos (shared kill rate across mice), Harris (shared rates across doses) |
| Hierarchical (population) fits (`hierarchical_joint_fit`) | mouse-to-mouse or patient-to-patient variation as random effects | LLC, LM2-4, Ghaffari Laleh |
| Two-population models (`lotka_volterra_competition`, `sensitive_resistant`, `null_coculture`) | interaction between cell lines; emerging resistance | Kaznatcheev co-culture (the paper's actual question), Bruchovsky PSA cycles |
| Dose-response laws (`theta_logistic_hill_kill`, `theta_logistic_hill_inhibition`, `pkpd_inhibition`, `transit_chain_erlang`, `damage_repair_arrest`) | Hill-type dose effects, drug PK, delayed onset | Harris erlotinib dose plates (the paper's DIP result) |
| Structural identifiability (`structural_identifiability_report`, `structural_identifiability`) | proof that a parameter can be identified from perfect data | any page; needs the optional StructuralIdentifiability backend |
| `practical_identifiability_report`, `global_sensitivity_analysis` | one combined report; which parameters drive the output | any page (the pieces it combines are shown) |
| `bootstrap_stage_uncertainty`, prediction bands | uncertainty carried through a staged fit | Ebos, Bruchovsky |
| `leave_one_out_validation`, `residual_analysis` | held-out error point by point, residual checks | any page |
| Staged workflows (`run_staged_pipeline`, `PipelineConfig`, `default_stages`) | build a model up stage by stage, locking earlier parameters | Ebos (growth first, then drug), Bruchovsky |
| Data checks and reports (`validate_timeseries`, `preflight_data_quality`, `generate_qc_report`) | schema and quality reports before fitting | every dataset's tidy table |
| Parameter sweeps (`run_sweep`) | what-if scenarios such as other dose schedules | Ebos, Bruchovsky |
| Control handoff (`controlled_model`, `mpc_functions`, `simulate_schedule`) | export a fitted model for optimal-control or MPC tools | Bruchovsky (adaptive therapy), Ebos |
| Other exposures (`PulseExposure`, `DecayingExposure`) | bolus dosing with drug decay, as in the Ebos/Benzekry 2024 model | Ebos |

## Gaps in the package itself

Building the showcase needed some code that the package does not provide, which suggests what to add next:

- **Batch fitting over many curves:** a call that fits a model panel to every curve in a table and returns the per-curve and pooled rankings (now `fit_panel` and `summarise_panel` in `showcase/analyses/common.jl`).
- **Selection rule as a function:** "fewest parameters within Δ BIC of the best" (now `rule_pick`).
- **A forecast check:** refit on the early part of each curve, score the rest (now `forecast_check`).
- **A profile search that steps outward from the estimate** and bisects the crossing, which was more reliable than a fixed grid on short curves (now `profile_intervals`).
- **Fixes found by running the package's own tools on real curves:**
    - `profile_likelihood`'s default grid spreads 8 points per side across the whole bound range. On these curves the nearest grid point is often far outside the interval, so only the estimate passes (a zero-width interval) or the interval runs to a bound.
    - Both cases are still labelled `bounded`: the check compares the last accepted point with the bound, and the log-spaced grid recomputes the bound with rounding error (1.0000000000000004e-6 > 1e-6).
    - `k_fold_cross_validation`'s default solver (`Rodas5`) differentiates the model in time, which fails on any model with a step exposure (a dose schedule); pass `solver = Tsit5()`.
    - `k_fold_cross_validation` starts each fold from the first training point, so a fold that holds out the first measurement predicts from the wrong initial condition.
- **Weighted errors in the joint-fit tools:** `run_joint_fit` and the tools built on it take one `residual_scale` per data set, not one per point. Proportional error needs the workaround in `package_checks.jl` (fit y/σ against u/σ). `k_fold_cross_validation` and `leave_one_out_validation` are unweighted and hold points out at random, so they measure interpolation, not forecasting.
- **Stochastic and Bayesian fitting:** Johnson et al. used stochastic birth-death models and several papers use Bayesian calibration; the package fits deterministic ODEs by least squares only.
