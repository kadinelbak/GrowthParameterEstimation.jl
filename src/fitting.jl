# Fitting module - Contains all ODE fitting and comparison functions
module Fitting

using StatsBase
using CSV
using DataFrames
using DifferentialEquations
using LsqFit
using RecursiveArrayTools
using DiffEqParamEstim
using Optimization
using ForwardDiff
using OptimizationOptimJL
using OptimizationBBO
using BlackBoxOptim

using ..Models

export setUpProblem, calculate_bic, pQuickStat, run_single_fit,
       compare_models, compare_datasets, compare_models_dict, fit_three_datasets

"""
    setUpProblem(model, x, y, solver, u0, p0, tspan, bounds; max_time=100.0, maxiters=10_000)

Set up and solve an ODE fitting problem using BlackBoxOptim. Returns the
optimized parameters, the dense solution, and the optimized problem.
"""
function setUpProblem(model, x, y, solver, u0, p0, tspan, bounds; max_time::Real = 100.0, maxiters::Integer = 10_000)
    prob = ODEProblem(model, u0, tspan, p0)

    loss = build_loss_objective(
        prob, solver,
        L2Loss(x, y),
        Optimization.AutoForwardDiff();
        maxiters = maxiters,
        verbose  = false,
    )

    result = bboptimize(
        loss;
        SearchRange = collect(zip(first.(bounds), last.(bounds))),
        Method      = :de_rand_1_bin,
        MaxTime     = float(max_time),
        TraceMode   = :silent,
    )

    p_opt, _ = best_candidate(result)
    prob_opt = remake(prob; p = p_opt, u0 = [y[1]], tspan = tspan)
    x_dense  = range(x[1], x[end], length = 1000)
    sol_opt  = solve(prob_opt, solver; reltol = 1e-12, abstol = 1e-12, saveat = x_dense)

    return p_opt, sol_opt, prob_opt
end

"""
    calculate_bic(prob, x, y, solver, p)

Compute BIC and SSR for a solved ODE model with parameters `p`.
"""
function calculate_bic(prob, x, y, solver, p)
    sol   = solve(remake(prob; p = p), solver; reltol = 1e-15, abstol = 1e-15, saveat = x)
    resid = y .- getindex.(sol.u, 1)
    ssr   = sum(resid .^ 2)
    k     = length(p)
    n     = length(x)
    bic   = n * log(ssr / n) + k * log(n)
    return bic, ssr
end

"""
    pQuickStat(x, y, p, sol, prob, bic, ssr)

Print a short summary of optimized parameters and fit statistics.
"""
function pQuickStat(x, y, p, sol, prob, bic, ssr)
    println("Optimized params: ", p)
    println("SSR: ", ssr)
    println("BIC: ", bic)
end

"""
    run_single_fit(x, y, p0; model=Models.logistic_growth!, fixed_params=nothing,
                   solver=Rodas5(), bounds=nothing, max_time=100.0, show_stats=true)

Fit a single model to `x`, `y` data with optional fixed parameters and bounds.
"""
function run_single_fit(
    x::Vector{<:Real},
    y::Vector{<:Real},
    p0::Vector{<:Real};
    model            = Models.logistic_growth!,
    fixed_params     = nothing,
    solver           = Rodas5(),
    bounds           = nothing,
    max_time::Real   = 100.0,
    show_stats::Bool = true,
)
    # Handle fixed parameters by wrapping the model
    if fixed_params !== nothing
        original_model = model
        n_total_params = length(p0) + length(fixed_params)

        model = function(du, u, p_free, t)
            p_full  = zeros(n_total_params)
            free_ix = 1
            for i in 1:n_total_params
                if haskey(fixed_params, i)
                    p_full[i] = fixed_params[i]
                else
                    p_full[i] = p_free[free_ix]
                    free_ix  += 1
                end
            end
            original_model(du, u, p_full, t)
        end

        free_indices = [i for i in 1:length(p0) if !haskey(fixed_params, i)]
        p0 = p0[free_indices]
        if bounds !== nothing
            bounds = bounds[free_indices]
        end
    end

    nparams = length(p0)

    _default_upper(p::Real) = max(10.0, abs(Float64(p)) * 100)

    if bounds === nothing
        bounds = [(0.0, _default_upper(p0[i])) for i in 1:nparams]
    else
        length(bounds) == nparams || throw(ArgumentError("bounds must have length $nparams"))
        bounds = [
            (
                isfinite(b[1]) ? Float64(b[1]) : 0.0,
                isfinite(b[2]) ? Float64(b[2]) : _default_upper(p0[i]),
            ) for (i, b) in enumerate(bounds)
        ]
    end

    x      = Float64.(x)
    y      = Float64.(y)
    tspan  = (x[1], x[end])
    u0     = [y[1]]

    p_opt, sol_opt, prob_opt = setUpProblem(model, x, y, solver, u0, p0, tspan, bounds; max_time = max_time)
    bic, ssr = calculate_bic(prob_opt, x, y, solver, p_opt)
    show_stats && pQuickStat(x, y, p_opt, sol_opt, prob_opt, bic, ssr)

    return (params = p_opt, bic = bic, ssr = ssr, solution = sol_opt)
end

"""
    compare_models(x, y, name1, model1, p0_1, name2, model2, p0_2; ...)

Fit two models to the same dataset and return a summary plus the best model.
"""
function compare_models(
    x::Vector{<:Real},
    y::Vector{<:Real},
    name1::String, model1::Function, p0_1::Vector{<:Real},
    name2::String, model2::Function, p0_2::Vector{<:Real};
    solver             = Rodas5(),
    bounds1            = nothing,
    bounds2            = nothing,
    fixed_params1      = nothing,
    fixed_params2      = nothing,
    show_stats::Bool   = false,
    output_csv::String = "model_comparison.csv",
)
    fit1 = run_single_fit(x, y, p0_1; model = model1, fixed_params = fixed_params1,
                          solver = solver, bounds = bounds1, show_stats = show_stats)
    fit2 = run_single_fit(x, y, p0_2; model = model2, fixed_params = fixed_params2,
                          solver = solver, bounds = bounds2, show_stats = show_stats)

    println("=== $name1 ===")
    println("Params: $(fit1.params), BIC: $(fit1.bic), SSR: $(fit1.ssr)")
    println("=== $name2 ===")
    println("Params: $(fit2.params), BIC: $(fit2.bic), SSR: $(fit2.ssr)")

    df_out = DataFrame(
        Model  = [name1, name2],
        Params = [string(fit1.params), string(fit2.params)],
        BIC    = [fit1.bic, fit2.bic],
        SSR    = [fit1.ssr, fit2.ssr],
    )
    CSV.write(output_csv, df_out)
    println("Results saved to $output_csv")

    best_model = fit1.bic <= fit2.bic ?
        (name = name1, params = fit1.params, bic = fit1.bic, ssr = fit1.ssr, solution = fit1.solution) :
        (name = name2, params = fit2.params, bic = fit2.bic, ssr = fit2.ssr, solution = fit2.solution)

    return (
        model1     = (name = name1, params = fit1.params, bic = fit1.bic, ssr = fit1.ssr, solution = fit1.solution),
        model2     = (name = name2, params = fit2.params, bic = fit2.bic, ssr = fit2.ssr, solution = fit2.solution),
        best_model = best_model,
    )
end

"""
    compare_datasets(x1, y1, name1, model1, p0_1, x2, y2, name2, model2, p0_2; ...)

Fit a model to two datasets and write a CSV summary.
"""
function compare_datasets(
    x1::Vector{<:Real}, y1::Vector{<:Real}, name1::String, model1::Function, p0_1::Vector{<:Real},
    x2::Vector{<:Real}, y2::Vector{<:Real}, name2::String, model2::Function, p0_2::Vector{<:Real};
    solver             = Rodas5(),
    bounds1            = nothing,
    bounds2            = nothing,
    fixed_params1      = nothing,
    fixed_params2      = nothing,
    show_stats::Bool   = false,
    output_csv::String = "dataset_comparison.csv",
)
    fit1 = run_single_fit(x1, y1, p0_1; model = model1, fixed_params = fixed_params1,
                          solver = solver, bounds = bounds1, show_stats = show_stats)
    fit2 = run_single_fit(x2, y2, p0_2; model = model2, fixed_params = fixed_params2,
                          solver = solver, bounds = bounds2, show_stats = show_stats)

    println("=== $name1 ===")
    println("Params: $(fit1.params), BIC: $(fit1.bic), SSR: $(fit1.ssr)")
    println("=== $name2 ===")
    println("Params: $(fit2.params), BIC: $(fit2.bic), SSR: $(fit2.ssr)")

    df_out = DataFrame(
        Dataset = [name1, name2],
        Params  = [string(fit1.params), string(fit2.params)],
        BIC     = [fit1.bic, fit2.bic],
        SSR     = [fit1.ssr, fit2.ssr],
    )
    CSV.write(output_csv, df_out)
    println("Results saved to $output_csv")
end

"""
    compare_models_dict(x, y, specs; default_solver=Rodas5(), show_stats=false, output_csv=\"all_models_comparison.csv\")

Fit each model in `specs` (a Dict of NamedTuples) to `x`, `y` and write summary tables.
"""
function compare_models_dict(
    x::Vector{<:Real},
    y::Vector{<:Real},
    specs::Dict{String,<:NamedTuple};
    default_solver        = Rodas5(),
    show_stats::Bool      = false,
    output_csv::String    = "all_models_comparison.csv",
)
    fits = Dict{String,Any}()
    results = NamedTuple[]

    for (name, spec) in specs
        solver_i = haskey(spec, :solver) ? spec.solver : default_solver
        fit = run_single_fit(
            x, y, spec.p0;
            model        = spec.model,
            fixed_params = get(spec, :fixed_params, nothing),
            solver       = solver_i,
            bounds       = get(spec, :bounds, nothing),
            show_stats   = show_stats,
        )
        fits[name] = fit
        push!(results, (Model = name, Params = fit.params, BIC = fit.bic, SSR = fit.ssr))
    end

    df_summary = DataFrame(
        Model  = [r.Model for r in results],
        Params = [string(r.Params) for r in results],
        BIC    = [r.BIC for r in results],
        SSR    = [r.SSR for r in results],
    )
    println("\nBIC Summary:")
    display(df_summary[:, [:Model, :BIC]])

    CSV.write(output_csv, df_summary)
    println("Summary saved to $output_csv")

    pred_rows = NamedTuple[]
    for (name, fit) in pairs(fits)
        for (t, u) in zip(fit.solution.t, fit.solution.u)
            push!(pred_rows, (Model = name, Time = t, Prediction = u[1]))
        end
    end
    df_preds = DataFrame(pred_rows)
    preds_csv = replace(output_csv, r"\\.csv$" => "_predictions.csv")
    CSV.write(preds_csv, df_preds)
    println("Predictions saved to $preds_csv")

    return fits
end

"""
    fit_three_datasets(x1, y1, name1, x2, y2, name2, x3, y3, name3, p0; ...)

Fit the same model to three datasets and return all results.
"""
function fit_three_datasets(
    x1::Vector{<:Real}, y1::Vector{<:Real}, name1::String,
    x2::Vector{<:Real}, y2::Vector{<:Real}, name2::String,
    x3::Vector{<:Real}, y3::Vector{<:Real}, name3::String,
    p0::Vector{<:Real};
    model                = Models.logistic_growth!,
    fixed_params         = nothing,
    solver               = Rodas5(),
    bounds               = nothing,
    show_stats::Bool     = false,
    output_csv::String   = "three_datasets_comparison.csv",
)
    fit1 = run_single_fit(x1, y1, p0; model = model, fixed_params = fixed_params,
                          solver = solver, bounds = bounds, show_stats = show_stats)
    fit2 = run_single_fit(x2, y2, p0; model = model, fixed_params = fixed_params,
                          solver = solver, bounds = bounds, show_stats = show_stats)
    fit3 = run_single_fit(x3, y3, p0; model = model, fixed_params = fixed_params,
                          solver = solver, bounds = bounds, show_stats = show_stats)

    println("=== $name1 ===")
    println("Params: $(fit1.params), BIC: $(fit1.bic), SSR: $(fit1.ssr)")
    println("=== $name2 ===")
    println("Params: $(fit2.params), BIC: $(fit2.bic), SSR: $(fit2.ssr)")
    println("=== $name3 ===")
    println("Params: $(fit3.params), BIC: $(fit3.bic), SSR: $(fit3.ssr)")

    df_out = DataFrame(
        Dataset = [name1, name2, name3],
        Params  = [string(fit1.params), string(fit2.params), string(fit3.params)],
        BIC     = [fit1.bic, fit2.bic, fit3.bic],
        SSR     = [fit1.ssr, fit2.ssr, fit3.ssr],
    )
    CSV.write(output_csv, df_out)
    println("Results saved to $output_csv")

    return (fit1 = fit1, fit2 = fit2, fit3 = fit3)
end

"""
    fit_three_datasets(x_datasets, y_datasets; p0=[0.1, 100.0], model=Models.logistic_growth!, fixed_params=nothing, solver=Rodas5(), bounds=nothing)

Convenience wrapper for fitting the same model to many datasets at once.
"""
function fit_three_datasets(
    x_datasets::Vector{Vector{<:Real}},
    y_datasets::Vector{Vector{<:Real}};
    p0::Vector{<:Real}     = [0.1, 100.0],
    model                  = Models.logistic_growth!,
    fixed_params           = nothing,
    solver                 = Rodas5(),
    bounds                 = nothing,
)
    n_datasets = length(x_datasets)
    @assert length(y_datasets) == n_datasets "Number of x and y datasets must match"

    individual_fits = Vector{Any}()
    for i in 1:n_datasets
        try
            fit_result = run_single_fit(
                x_datasets[i], y_datasets[i], p0;
                model        = model,
                fixed_params = fixed_params,
                solver       = solver,
                bounds       = bounds,
                show_stats   = false,
            )
            push!(individual_fits, (dataset = i, fit_result = fit_result))
        catch e
            println("Warning: Failed to fit dataset $i: $e")
            push!(individual_fits, (dataset = i, fit_result = nothing))
        end
    end

    successful_fits = filter(f -> f.fit_result !== nothing, individual_fits)

    if !isempty(successful_fits)
        all_params = [f.fit_result.params for f in successful_fits]
        mean_params = [mean([p[i] for p in all_params]) for i in 1:length(all_params[1])]
        std_params  = [std([p[i] for p in all_params]) for i in 1:length(all_params[1])]
        mean_ssr    = mean([f.fit_result.ssr for f in successful_fits])

        summary = (
            mean_params   = mean_params,
            std_params    = std_params,
            mean_ssr      = mean_ssr,
            n_successful  = length(successful_fits),
            n_total       = n_datasets,
        )
    else
        summary = (
            mean_params   = Float64[],
            std_params    = Float64[],
            mean_ssr      = Inf,
            n_successful  = 0,
            n_total       = n_datasets,
        )
    end

    return (individual_fits = individual_fits, summary = summary)
end

end # module Fitting
