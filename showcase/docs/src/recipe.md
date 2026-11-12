# Shared recipe

The same steps run on every dataset (`showcase/analyses/common.jl`).

## Error model

Proportional error with a floor: ``\sigma_i = 0.1\,\max(y_i,\ 0.05\,\max_j y_j)``. The weighted residual sum of squares is

```math
\mathrm{WSSE} = \sum_i \left(\frac{y_i - \hat y_i}{\sigma_i}\right)^2 .
```

The first observation is the initial condition, so no initial size is estimated.

## Growth panel

| Model | Equation | k |
| --- | --- | --- |
| exponential | ``\dot V = aV`` | 1 |
| logistic | ``\dot N = rN(1 - N/K)`` | 2 |
| Gompertz | ``\dot N = rN\ln(K/N)`` | 2 |
| theta-logistic | ``\dot N = rN(1 - (N/K)^\theta)`` | 3 |
| Bertalanffy | ``\dot V = aV^{2/3} - bV`` | 2 |
| general Bertalanffy | ``\dot V = aV^\gamma - bV`` | 3 |
| power law | ``\dot V = aV^\gamma`` | 2 |
| exponential-linear | ``\dot V = a_0V / (1 + (a_0V/a_1)^{\psi})^{1/\psi}``, ``\psi = 20`` | 2 |

## Fitting

Each curve is fitted separately with `fit(spec, [condition]; n_starts = 8, optimizer = :transformed)`: eight starts (the box midpoint plus seven random points, drawn on a log scale for parameters whose bounds span several decades), each optimised with BFGS on an unconstrained transform of the bounded parameters.

## Scores and the selection rule

With ``n`` points and ``k`` parameters, the package reports ``\mathrm{AIC} = n\ln(\mathrm{WSSE}/n) + 2k`` and ``\mathrm{BIC} = n\ln(\mathrm{WSSE}/n) + k\ln n``; AICc adds ``2k(k+1)/(n-k-1)``. Dataset totals are sums over curves.

**Rule:** among models within 2 BIC of the lowest, pick the one with the fewest parameters (ties go to the lower BIC). The rule is applied per curve and to the dataset totals.

## Profile intervals

For each parameter of the pick, the parameter is held fixed (`fit(...; fixed_params)`) and the others are refitted. Starting at the estimate, the search steps outward with growing steps (on a log scale for positive parameters) until ``n\ln(\mathrm{WSSE}/n)`` rises by 3.84, the χ²₁ 95% cutoff, then bisects the crossing. An interval that reaches a bound is reported as open (not identifiable on that curve).

## Forecast check

Every model is refitted on the first 70% of each curve's points and asked to predict the rest. The page reports the median relative error on those held-out points and the share of predictions within 20% of the data.
