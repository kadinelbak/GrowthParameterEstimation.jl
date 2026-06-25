# Getting Started

This tutorial will walk you through the basics of using GrowthParameterEstimation.jl to fit growth models to experimental data.

## Loading the Package

First, load the package into your Julia session:

```julia
using GrowthParameterEstimation
```

## Preparing Data

Let's create some synthetic growth data to work with:

```julia
# Time points
t = 0.0:0.5:10.0

# Synthetic logistic growth data with measurement noise
y = [100.0 / (1.0 + 99.0 * exp(-0.5 * ti)) for ti in t] + 0.5*randn(length(t))

# Initial condition
u0 = [y[1]]

# Initial parameter guess [growth rate, carrying capacity]
p0 = [0.3, 50.0]
```

## Fitting a Model

Now let's fit a logistic growth model to our data:

```julia
# Fit the model using the built-in logistic growth builder
result = run_single_fit(t, y, p0; model=Models.build_logistic())

# Examine the results
println("Optimized growth rate: $(result.params[1])")
println("Optimized carrying capacity: $(result.params[2])")
println("BIC: $(result.bic)")
println("SSR: $(result.ssr)")
```

## Visualizing Results

Let's plot the data and the fitted model:

```julia
using Plots  # You'll need to install this separately if you don't have it

# Generate predictions from the fitted model
t_fine = 0.0:0.1:12.0
pred = predict_model(Models.build_logistic(), t, result.params, 0.0, y[1]; n_curve=length(t_fine))

# Create the plot
plot(t, y, seriestype=:scatter, label="Data", xlabel="Time", ylabel="Population")
plot!(t_fine, pred[2], label="Fitted Model", linewidth=2)
```

## Comparing Models

Let's compare the logistic growth model with an exponential growth model:

```julia
# Define an exponential growth model
exponential_growth!(du, u, p, t) = du[1] = p[1] * u[1]

# Compare the models
comparison = compare_models(
    t, y,
    "logistic", Models.build_logistic(), [0.3, 50.0],
    "exponential", exponential_growth!, [0.1]
)

# See which model is better
println("Logistic BIC: $(comparison.model1.bic)")
println("Exponential BIC: $(comparison.model2.bic)")
println("Best model: $(comparison.best_model.name)")
```

## Next Steps

You've now seen the basic workflow for fitting and comparing growth models. To learn more:

1. Check out the `basic_examples.md` tutorial for more common usage patterns
2. Explore the `advanced_usage.md` tutorial for complex workflows
3. Look at the API documentation for detailed information on all available functions