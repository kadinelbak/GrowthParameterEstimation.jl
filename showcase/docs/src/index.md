# GrowthParameterEstimation.jl showcase

This site applies one package, one fitting recipe and one model-selection rule to published, public tumour-growth datasets: cells in a dish, spheroids, mice and patients.

Each dataset page shows the same things in the same order:

1. **Model comparison**: every growth law in the panel fitted to every curve, scored by BIC, AIC and AICc, with the pick under the rule *fewest parameters within 2 BIC of the best*.
2. **Identifiability**: profile-likelihood 95% intervals for the picked model on every curve.
3. **Forecast check**: refit on the early part of each curve, predict the rest.
4. **Reproduce**: the two commands that rebuild the page from the original source files.

## Why one package

The original papers each wrote their own fitting code, error model and selection rule, so their conclusions are hard to compare. Here the calls are the same everywhere (`get_model`, `fit`, `simulate` and a profile built from `fit(...; fixed_params)`), so differences between datasets come from the data, not the code.

## Data policy

No third-party data is stored in this repository. `showcase/data/prepare.py` downloads each file from its original source, checks its SHA-256 hash and writes tidy tables locally. Only derived results (fitted parameters, rankings, figures) are committed. Every page cites the original paper and states the source licence.
