# paper.md outline: Statement of need and State of the field

Bullets only, for the author to turn into prose. Items marked **[check]** need
the author to confirm or supply a citation before they go into `paper.md`.
Nothing here should be claimed in the paper without the pointer next to it.

## Statement of need

- **Problem.** Fitting mechanistic ODE models to in-vitro cell-count time
  series is rarely one fit. A real study fits a sequence of related experiments
  (untreated growth, then nutrient or drug perturbations, then combinations,
  then cocultures), and each stage should reuse what the previous stage
  estimated.
- **What researchers need at each stage.**
  - Several candidate models compared on the same data with a stated selection
    rule (BIC; fewest parameters within a ΔBIC threshold of the best).
  - Joint fits of one multi-state model across many wells and conditions, with
    shared and condition-specific parameters.
  - Initial states that can be estimated when the first count is far from the
    seeding.
  - Multistart optimisation, because these likelihoods are multimodal.
  - Uncertainty and identifiability answers for the chosen model: bootstrap,
    profile likelihood, Fisher information, synthetic recovery, structural
    checks.
- **What the package gives.** One Julia package that covers all of the above,
  plus a staged pipeline that carries parameters from stage to stage, schema
  validation of the input table, run manifests and resume-from-stage
  (README "Features").
- **Who it is for.** Mathematical oncology and quantitative cell biology
  groups fitting growth, treatment-response and resistance models to plate or
  live-imaging counts. Not limited to cancer: any growth ODE with dose and
  condition metadata **[check: do you want to claim this beyond cancer? The
  thesis draft says the package "is intended to be adaptable and extend beyond
  ovarian cancer growth modeling"]**.
- **Evidence of use (point to `docs/src/research_use.md`).**
  - A2780 / A2780cis cisplatin resistance and dose scheduling: every candidate
    fit goes through `run_joint_fit`; held-out validation uses
    `run_joint_multistart`; results feed an optimal-control study. Dependency
    since 2026-02-08. Public results site.
  - OVCAR3 olaparib resistance and glutamine starvation: multistart model
    comparison over 16 transition models, profile-likelihood and Fisher
    intervals, bootstrap and synthetic recovery, all from the package.
  - Honors thesis (UF BME) and an A2780 manuscript, both in preparation
    **[check: how you want to cite them; JOSS reviewers can't open private
    repos, so the public results site and the thesis are the strongest
    pointers]**.

## State of the field

Group the alternatives and say, for each group, what it does well and what the
package adds. **[check]** every row against the current version of each tool.

- **General Julia SciML stack** (DiffEqParamEstim.jl, Optimization.jl,
  DifferentialEquations.jl). The package is built on these.
  - Strength: fast, flexible solvers and optimisers.
  - Gap the package fills: the user writes the loss, multistart, model
    comparison, staging and uncertainty code by hand.
- **Julia parameter-estimation and identifiability tools** (PEtab.jl,
  LikelihoodProfiler.jl, StructuralIdentifiability.jl).
  - Strength: rigorous estimation and profiling for systems-biology models;
    StructuralIdentifiability.jl is used inside the package for symbolic checks.
  - Gap: no growth-model library, no condition/dose/cell-line metadata table,
    no staged inheritance between related experiments **[check PEtab.jl's
    current features before claiming this]**.
- **Growth-curve fitting packages** (R: growthcurver, Sprouffske & Wagner 2016;
  grofit, Kahm et al. 2010; growthrates).
  - Strength: quick, well-tested fits of one curve with a fixed growth law.
  - Gap: single-curve, single-population; no multi-state ODEs, no joint fits
    across conditions, no drug-response stages.
- **Systems-biology estimation suites outside Julia** (Data2Dynamics in MATLAB;
  pyPESTO and AMICI in Python; COPASI).
  - Strength: mature profile likelihood and multistart.
  - Gap: model specification in SBML or another language, and no growth- or
    treatment-specific workflow **[check: say this neutrally; these tools are
    powerful]**.
- **Context from the modelling literature** (to motivate model comparison
  rather than one fixed growth law):
  - Benzekry et al. 2014, PLoS Computational Biology, comparison of classical
    tumour growth models.
  - Gerlee 2013, Cancer Research, "The model muddle".
  - Strobl et al. on resistance and adaptive therapy (the A2780 pipeline has a
    `strobl_models.jl` family) **[check which Strobl paper]**.
- **Build-vs-contribute justification** (JOSS asks for this): why not add the
  workflow to an existing package. Possible points **[check]**:
  - The staged, metadata-driven workflow is an application layer, not a
    solver or estimator, so it sits on top of SciML rather than inside it.
  - Growth-curve packages are single-curve by design.

## Things to fix in the drafts before reusing their text

- The manuscript bullet draft and the thesis drafts say models are selected
  "within 3 BIC"; the A2780 pipeline (`zoo_analyses.jl`) and the OVCAR3
  pipeline (`[selection] delta_bic`) use 2 BIC. Pick one and say it the same
  way in the paper.
- The class outline promises code "[DOI]" for both repositories; a Zenodo DOI
  for the package will be needed for JOSS anyway.
