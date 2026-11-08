# GrowthParameterEstimation.jl

```@meta
CurrentModule = GrowthParameterEstimation
```

`GrowthParameterEstimation.jl` is a Julia package for fitting growth ordinary
differential equation (ODE) models to time-series data. It provides built-in and
user-registered growth models, single and joint (shared-parameter) fitting,
model comparison by information criteria, staged pipelines that pass parameters
from one experiment to the next, uncertainty quantification, and practical and
structural identifiability analysis.

Release notes for every version, including breaking changes, are in
[`CHANGELOG.md`](https://github.com/kadinelbak/GrowthParameterEstimation.jl/blob/main/CHANGELOG.md).

## Quick example

```julia
using GrowthParameterEstimation

x = collect(0.0:1.0:5.0)
y = [5.0, 8.0, 12.0, 18.0, 22.0, 28.0]
result = run_single_fit(x, y, [0.2, 50.0]; show_stats = false)
result.params, result.bic
```

## Where to go next

- [Installation](installation.md)
- [Getting Started](tutorial/getting_started.md)
- [Basic Examples](tutorial/basic_examples.md)
- [Advanced Usage](tutorial/advanced_usage.md)
- [Identifiability](identifiability.md)
- [API Reference](api.md)

Runnable templates live in the repository:

- `examples/pipeline_one_shot_template.jl`: end-to-end staged pipeline script.
- `tests/notebooks/function_tour.ipynb`: walkthrough of the public API.
- `tests/notebooks/pipeline_step_by_step_template.ipynb`: staged pipeline, one step at a time.

## Getting help and contributing

Questions, bug reports and feature requests go to the
[issue tracker](https://github.com/kadinelbak/GrowthParameterEstimation.jl/issues).
See [`CONTRIBUTING.md`](https://github.com/kadinelbak/GrowthParameterEstimation.jl/blob/main/CONTRIBUTING.md)
for how to set up a development environment, run the tests and submit changes.
