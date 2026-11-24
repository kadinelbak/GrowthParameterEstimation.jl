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

## Not shown yet

| Package feature | What it would add | Natural dataset |
| --- | --- | --- |
| Joint fits with shared parameters (`fit(...; shared_params, tie_constraints)`, `run_joint_fit`) | one drug effect estimated from many weak curves instead of one per curve | Ebos (shared kill rate across mice), Harris (shared rates across doses) |
| Hierarchical (population) fits (`hierarchical_joint_fit`) | mouse-to-mouse or patient-to-patient variation as random effects | LLC, LM2-4, Ghaffari Laleh |
| Two-population models (`lotka_volterra_competition`, `sensitive_resistant`, `null_coculture`) | interaction between cell lines; emerging resistance | Kaznatcheev co-culture (the paper's actual question), Bruchovsky PSA cycles |
| Dose-response laws (`theta_logistic_hill_kill`, `theta_logistic_hill_inhibition`, `pkpd_inhibition`, `transit_chain_erlang`, `damage_repair_arrest`) | Hill-type dose effects, drug PK, delayed onset | Harris erlotinib dose plates (the paper's DIP result) |
| Built-in identifiability tools (`profile_likelihood`, `fisher_information`, `structural_identifiability_report`, `practical_identifiability_report`, `synthetic_recovery_benchmark`, `global_sensitivity_analysis`) | structural checks before fitting, Fisher intervals, recovery tests on synthetic data | any page; the showcase currently uses its own profile loop |
| Uncertainty by resampling (`bootstrap_joint_fit`, `bootstrap_stage_uncertainty`) | parameter and prediction bands | any page |
| Cross-validation (`leave_one_out_validation`, `k_fold_cross_validation`) and `residual_analysis` | held-out error by curve, residual checks | any page |
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
- **Stochastic and Bayesian fitting:** Johnson et al. used stochastic birth-death models and several papers use Bayesian calibration; the package fits deterministic ODEs by least squares only.
