---
title: 'GrowthParameterEstimation.jl: staged, identifiability-aware fitting of growth models to cell-population time series'
tags:
  - Julia
  - ordinary differential equations
  - parameter estimation
  - model selection
  - identifiability
  - mathematical oncology
authors:
  - given-names: Kadin
    surname: El Bakkouri
    orcid: 0000-0000-0000-0000 # TODO: add ORCID
    affiliation: 1
affiliations:
  - index: 1
    name: J. Crayton Pruitt Family Department of Biomedical Engineering, University of Florida, United States
    ror: 02y3ad647
date: 9 November 2026
bibliography: paper.bib
---

# Summary

Growth experiments in cell biology measure how many cells there are over time,
under different seeding densities, drugs, doses and nutrient conditions.
Mechanistic ordinary differential equation (ODE) models turn those counts into
interpretable quantities such as growth rates, carrying capacities, drug kill
rates and switching rates between cell states. `GrowthParameterEstimation.jl`
is a Julia [@bezanson2017julia] package that fits such models to time-series
counts, compares candidate models, and reports how well each parameter is
determined by the data. It provides a library of growth laws, a registry for
user-defined multi-state models, joint fitting of one model across many wells
and conditions, a staged pipeline that carries estimates from simpler
experiments into more complex ones, and uncertainty and identifiability tools
(bootstrap, profile likelihood, Fisher information, synthetic recovery and
symbolic structural checks). It is built on the SciML ecosystem
[@rackauckas2017differentialequations], is registered in the Julia General
registry, and is archived on Zenodo [@elbakkouri2026growthparameterestimation].

# Statement of need

A realistic growth study is rarely a single curve fit. A typical experiment
first measures untreated growth, then adds a drug or removes a nutrient, then
combines perturbations, and sometimes mixes sensitive and resistant
populations in coculture. Each later experiment should reuse what the earlier
ones estimated, or the larger models become unidentifiable. At every step the
analyst also has to choose between competing mechanisms, fit one multi-state
model jointly across many wells, estimate initial states when the first count
is far from the seeding, restart the optimiser from many points because the
likelihood is multimodal, and check whether the selected model's parameters
are actually determined by the data.

In practice this work is usually done with hand-written scripts around a
general ODE solver and optimiser. The loss function, multistart logic,
information criteria, stage bookkeeping and uncertainty code are rewritten for
each project, which makes analyses hard to reproduce and to compare.
`GrowthParameterEstimation.jl` packages that workflow. Users supply a tidy
table of times, counts and condition metadata (dose, cell line, density,
replicate). The package validates the table, fits the requested candidate
models, ranks them by the Bayesian information criterion (BIC)
[@schwarz1978estimating; @kass1995bayes], and exports tables, figures and a
run manifest that allows resuming from any stage.

The target users are mathematical oncology and quantitative cell biology
groups fitting growth, treatment-response and resistance models to plate or
live-imaging counts. The package was written for ovarian cancer drug
resistance, but nothing in the API is specific to cancer. Any growth ODE with
dose and condition metadata can be registered and fitted.

# State of the field

Model choice matters for growth data: classical tumour growth laws fitted to
the same measurements can give very different parameters and predictions
[@benzekry2014classical; @gerlee2013model]. That motivates fitting and
comparing several candidate models as a matter of routine. Existing tools cover
parts of this workflow.

- **The Julia SciML stack.** DifferentialEquations.jl
  [@rackauckas2017differentialequations], Optimization.jl
  [@dixit2023optimization] and Optim.jl [@mogensen2018optim] provide fast
  solvers and optimisers, and `GrowthParameterEstimation.jl` is built on them.
  They are deliberately general, so the user still writes the loss, the
  multistart, the model comparison and the stage handling.
- **Estimation and identifiability tools for systems biology.** PEtab.jl
  [@persson2025petab], LikelihoodProfiler.jl [@borisov2020confidence] and
  StructuralIdentifiability.jl [@dong2023differential] in Julia, and
  Data2Dynamics [@raue2015data2dynamics], pyPESTO [@schalte2023pypesto] and
  COPASI [@hoops2006copasi] elsewhere, provide rigorous estimation and
  profiling. They expect models written in a formal specification such as
  SBML or PEtab, and they have no notion of a growth-model library or of
  carrying parameters between related experiments.
  `GrowthParameterEstimation.jl` uses StructuralIdentifiability.jl internally
  for its symbolic checks.
- **Growth-curve packages.** R packages such as growthcurver
  [@sprouffske2016growthcurver] and grofit [@kahm2010grofit] give quick,
  well-tested fits of a single curve to a fixed growth law. They are designed
  for one curve and one population, without multi-state ODEs, joint fits across
  conditions or drug-response stages.

The package was built as an application layer on top of these tools rather
than as a contribution to one of them. The staged, metadata-driven workflow
sits above the level of a solver or estimator, and growth-curve packages are
single-curve by design, so adding it to either would change what those
projects are for.

# Software design

The package is organised around four decisions, each a trade-off.

**Composable model specifications.** Models are `ModelSpec` objects with named
parameters, bounds, a number of states and an observable, built either from
builder functions (`build_logistic`, `build_gompertz` and others) or written by
the user and added with `register_model!`. Earlier versions exposed raw
right-hand-side functions. These were replaced by the registry so that every
model carries its own parameter names and bounds, and so that a new mechanism
can be added without forking the package. The cost is a small amount of extra
boilerplate for one-off models.

**Staged fitting with inherited parameters.** `run_staged_pipeline` fits a
sequence of `PipelineStage`s. Each stage can fix parameters to values estimated
in earlier stages, for example taking the growth rate and carrying capacity
from untreated wells before fitting kill parameters to treated wells. This
shrinks the search space at each stage and keeps large models identifiable, at
the price of ignoring the uncertainty in inherited values. The alternative, one
joint fit of every parameter to every experiment, is available through
`run_joint_fit` and can be used to check the staged result. Stages can run
automatically (best model by BIC) or from a checkpoint with a model chosen by
hand, because the scientifically right model is not always the BIC winner.

**Joint fits with explicit pooling.** `run_joint_fit` fits one model across
many trajectories, with each parameter declared shared, per-condition or fixed,
and with optional estimation of initial states. `hierarchical_joint_fit` adds
partial pooling across groups. BIC counts only the parameters that are
actually optimised, so fixed seeding states do not penalise a model. Fits use
bounded quasi-Newton optimisation (BFGS inside Fminbox, from Optim.jl) with
Nelder–Mead as a fallback, started from many points (`generate_multistarts`,
`run_joint_multistart`). Non-finite or negative solutions are rejected rather
than silently ranked. Local methods from many starts were preferred to a single
global search because they are faster per start, easier to diagnose, and make
multimodality visible.

**Identifiability as a separate gate.** Selecting a model by BIC does not
guarantee that its parameters are determined by the data. The package therefore
keeps uncertainty and identifiability analysis beside model selection rather
than folding it in: `bootstrap_joint_fit` for resampled intervals,
`profile_likelihood` [@raue2009structural] for the selected models,
`fisher_information` for a fast local screen, `synthetic_recovery_benchmark`
to check whether known parameters can be recovered, and
`structural_identifiability` for symbolic checks. Global sensitivity analysis
is framed as an exploration of the stated parameter ranges, not as a posterior
analysis. The cheap tools (Fisher information) can screen every candidate,
while the expensive ones (profiles, bootstrap) are reserved for the models that
survive selection.

Generic fitting, pooling and diagnostics live in the package. Biology-specific
equations, stage definitions and reports stay in the research projects that use
it. A control handoff API (`controlled_model`) exports a fitted model, with its
bootstrap parameter sets, to external optimal-control tools without solving
control problems inside the package.

# Research impact statement

The package is the fitting engine of two ongoing research projects on drug
resistance in ovarian cancer cell lines, documented with file-level pointers in
the documentation's Research use page.

- **A2780 cisplatin resistance and dose scheduling.** Joint fits of a family of
  compartment models to monoculture and coculture time-lapse counts of
  cisplatin-sensitive (A2780) and cisplatin-resistant (A2780cis) cells, with
  multistart held-out validation. The selected model feeds an optimal-control
  study of cisplatin schedules. The package has been a dependency of this
  project since February 2026, and its results are public at
  <https://kadinelbak.github.io/CancerGrowthDynamics/>.
- **OVCAR3 olaparib resistance and glutamine starvation.** Multistart
  comparison of 16 transition models between sensitive and resistant states,
  with profile-likelihood and Fisher intervals, bootstrap and synthetic
  recovery, all from the package. This extends experimental work on metabolic
  plasticity and PARP-inhibitor persistence [@delpinoherrera2025metabolic].
  [TODO: confirm whether the posted or revised preprint uses the package.]
- **Benchmark across public datasets.** The repository includes a showcase that
  refits published tumour and cell-growth datasets from other groups with the
  same workflow and compares growth laws across them. [TODO: update with the
  dataset count and link once the showcase is merged.]

These analyses support an undergraduate honors thesis in biomedical
engineering at the University of Florida and manuscripts in preparation.

# AI usage disclosure

Generative AI tools were used in developing this software, its documentation
and this paper. The author wrote the original fitting code in 2025 and a full
staged fitting pipeline by hand in early 2026 (v0.3.0), using AI only for
polishing. From April to 22 August 2026, OpenAI Codex (GPT-5.3 and GPT-5.4)
restructured that pipeline into the current modular package at the author's
request. In that period Codex wrote the large implementation commits,
including joint fitting (v0.4) and the identifiability suite. It also drafted
the feature specifications in `context/feature-specs/`, which the author edited
before implementation. From 22 August 2026, Claude Code (Claude Opus 4.7, then
Claude Opus 5.5) was used for later changes, including the v0.5 releases, the
control handoff API, the citation metadata, documentation and repository
cleanup, and dependency updates. Short trials of opencode and Continue.dev with
an NVIDIA Nemotron model in June 2026 were mostly unsuccessful. opencode did a
small amount of refactoring, which was largely corrected with Claude Code in
August 2026. `docs/AI Reference.md` is a guide
for users who integrate the package with an AI assistant, not AI-written code.
Claude Code drafted the prose of this paper from the author's outline, and the
author revised and approved the text.

The author made the primary design decisions described above. These are the
split between the generic package and project-specific code, BIC-based
selection, staged inheritance, explicit pooling, the identifiability methods,
and removing an earlier graphical interface in favour of a programmatic API.
AI-generated code was checked by the package test suite (about 250 tests)
under continuous integration on Julia 1.10, 1.11 and 1.12, by use in the
author's own analyses, and by scientific review of any change to model rankings
before merging. [TODO: add any hand checks, e.g. fits against known parameters
or another tool.] The author reviewed, edited and tested all AI-assisted
content and takes responsibility for its accuracy and licensing. No AI was used
to write replies to editors or reviewers.

# Acknowledgements

The author thanks Dr. Adriana Del Pino Herrera for teaching the modelling
foundations this work builds on, and Dr. Meghan Ferrall-Fairbanks and the BEAT
Cancer Lab for support. [TODO: funding statement with grant numbers, or "This
work received no specific external funding."] The author declares no competing
interests.

# References
