# Why one package

Each paper behind these datasets wrote its own fitting code, with its own error model, its own optimiser and its own way of choosing a model. Their conclusions are sound on their own terms, but they cannot be lined up against each other: when two papers favour different growth laws, it is unclear whether the tumours differ or the analyses do.

This showcase removes the second explanation. Every dataset goes through the same calls, the same error model and the same rule, so a difference between two dataset pages comes from the data.

## The whole analysis in a few calls

For one curve, every step on a dataset page is a call into the package:

```julia
using GrowthParameterEstimation

c = FitCondition("mouse 1", t, y, sigma, [y[1]], ConstantExposure(0.0), Dict{Symbol,Any}())

fits = Dict(m => fit(get_model(m), [c]; n_starts = 8, optimizer = :transformed)
            for m in ["exponential_growth", "logistic_growth", "classic_gompertz_growth", "power_law_growth"])

bic  = Dict(m => r.best.bic for (m, r) in fits)          # AIC and the parameters sit next to it
best = fits["classic_gompertz_growth"]

# profile likelihood: hold r fixed, refit the rest
prof = fit(get_model("classic_gompertz_growth"), [c]; fixed_params = Dict(:r => 0.2), optimizer = :transformed)

# prediction from the fitted parameters
sim = simulate(get_model("classic_gompertz_growth"), collect(range(t[1], 2t[end]; length = 100)),
               best.best.params; u0 = [y[1]])
```

`showcase/analyses/common.jl` wraps exactly these calls in loops over curves and models, and `showcase/run_pipeline.jl` runs that over every dataset in `pipeline.toml`.

## What bringing the datasets together buys

- **One scale for every result.** Cells in a dish, spheroids and mouse tumours are scored with the same criteria, so the [Across datasets](datasets/cross.md) page can put them on one heatmap.
- **The same honesty checks everywhere.** Every pick gets profile-likelihood intervals and a forecast test, including on datasets whose original paper reported neither.
- **Per-curve and pooled answers side by side.** Many curves are too short to show saturation and prefer exponential growth on their own, even when the pooled data clearly favour a saturating law. Seeing both on every page keeps that from being mistaken for a biological difference.
- **New data costs one block of TOML.** Point `pipeline.toml` at a CSV with a curve id, a time and a value column, and the same pages are produced for your own experiment (see [Reproduce](reproduce.md)).
- **Speed.** Each dataset page, including profiles on every curve and the forecast check, rebuilds in minutes on a 4-core laptop. The run time is recorded in each `results/<id>/meta.toml`.

## What this does not replace

The showcase uses classic single-population laws and the first observation as the initial size. Several source papers fitted richer, dataset-specific models (necrotic-core spheroid models, Allee effects, drug terms). Dataset pages say where the setup differs from the paper, and their rankings are re-analyses with a shared recipe, not reproductions of the published numbers.
