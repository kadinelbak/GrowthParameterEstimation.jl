# Contributing to GrowthParameterEstimation.jl

Thank you for your interest in improving GrowthParameterEstimation.jl. Bug
reports, questions, documentation fixes and code contributions are all
welcome. By taking part you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Getting help and reporting problems

- **Questions and usage help:** open an issue on the
  [issue tracker](https://github.com/kadinelbak/GrowthParameterEstimation.jl/issues)
  and describe what you are trying to do.
- **Bug reports:** use the "Bug report" issue template. Include your Julia
  version (`versioninfo()`), the package version (`Pkg.status()`), a minimal
  script that reproduces the problem and the full error message.
- **Feature requests:** use the "Feature request" template and explain the
  modelling or analysis problem the feature would solve.

## Setting up a development environment

```bash
git clone https://github.com/kadinelbak/GrowthParameterEstimation.jl.git
cd GrowthParameterEstimation.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The package supports Julia 1.10 and later (see `Project.toml`).

## Running the tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

All tests must pass before a pull request is merged. Continuous integration
runs the same suite on Julia 1.10, 1.11 and 1.12.

## Building the documentation

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The built site is written to `docs/build/` (ignored by git). New public
functions should have a docstring; it appears automatically on the API
Reference page.

## Submitting changes

1. Fork the repository and create a branch from `main`.
2. Make your change, with tests for new behaviour or bug fixes.
3. Format code with [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl)
   using the repository's `.JuliaFormatter.toml`.
4. Add a short entry to `CHANGELOG.md` under an "Unreleased" heading.
5. Open a pull request describing what changed and why. Link any related issue.

Please do not commit generated outputs (figures, CSV results, PDFs); the
`output/` and `tests/outputs/` folders are ignored for that reason.

## Releases

Versions follow [semantic versioning](https://semver.org/) (pre-1.0: a minor
bump may contain breaking changes, which are listed in `CHANGELOG.md`).
Releases are registered in the Julia General registry with
[Registrator](https://github.com/JuliaRegistries/Registrator.jl), and TagBot
creates the matching GitHub release.
