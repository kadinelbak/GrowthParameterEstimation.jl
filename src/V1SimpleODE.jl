module V1SimpleODE

using StatsBase
using CSV
using Plots
using DataFrames
using DifferentialEquations
using SciMLSensitivity
using LsqFit
using RecursiveArrayTools
using DiffEqParamEstim
using Optimization
using ForwardDiff
using OptimizationOptimJL
using OptimizationBBO
using BlackBoxOptim
using Statistics

"""
    extract_day_averages_from_df(dfTemp)

Extracts non-missing values from a DataFrame and assigns time indices.
Returns two Float64 arrays: time points `x` and values `y`. Note that names 
for lines need to follow this pattern <celltype>_<drug_concentration>_<treated/untreated>_<Day#>_<Tile-#>_<Well/Sample>.
example: `A2780cis_15and20__Treated_Day1_Tile-1_A5`.
"""
function extract_day_averages_from_df(df::DataFrame)
    # 1) keep only the Tile rows you care about
    df = filter(row -> occursin(r"_Tile-\d+_[^AC]\d", row.Image), df)

    # 2) pull out the day number
    extract_day(name::AbstractString) = begin
        m = match(r"(?i)day(\d+)", name)
        m !== nothing ? parse(Int, m.captures[1]) : missing
    end
    df.day = extract_day.(df.Image)
    df = dropmissing(df, :day)

    # 3) group by day, chunk into 18-tile batches, compute means
    grouped = groupby(df, :day)
    new_rows = Vector{NamedTuple{(:Day, :Average), Tuple{Int,Float64}}}()
    for g in grouped
        for i in 1:18:nrow(g)
            chunk = g[i : min(i+17, nrow(g)), :]
            avg = mean(chunk[!, Symbol("Area µm^2")])
            push!(new_rows, (Day = unique(chunk.day)[1], Average = avg))
        end
    end

    # 4) assemble and show the small DataFrame
    df_avg = DataFrame(new_rows)
    println("This is what the data looks like:\n", df_avg)

    # 5) turn into plain Float64 vectors
    x = Float64.(df_avg.Day)
    y = Float64.(df_avg.Average)

    return x, y
end


"""
    extractData(file_name)

    Takes a data file that has one column. One column that has a header of "Day Averages"
"""

function extractData(file_name)
    df = CSV.read(file_name, DataFrame)

    x = []
    y = []
    current_day = 1
    for row in eachrow(df)
        val = row[:"Day Averages"]
        if !ismissing(val) && !isempty(strip(string(val)))  # Filters out missing or blank cells
            push!(x, current_day)
            push!(y, val)  # No need to parse
            current_day += 1
        end
    end

    x = Float64.(x)
    y = Float64.(y)
    
    return x, y
end


"""
    setUpProblem(model, xdata, ydata, solver, u0, p, tspan, bounds)

Sets up and solves an ODE fitting problem using BlackBoxOptim.
Returns optimized parameters, solution, and the problem object.
"""
function setUpProblem(model, x, y, solver, u0, p0, tspan, bounds)
    prob = ODEProblem(model, u0, tspan, p0)
    solve(prob, solver, saveat=x, reltol=1e-16, abstol=1e-16)

    loss = build_loss_objective(
        prob, solver,
        L2Loss(x, y),
        Optimization.AutoForwardDiff();
        maxiters=10_000, verbose=false
    )

    result = bboptimize(
        loss;
        SearchRange = collect(zip(first.(bounds), last.(bounds))),
        Method      = :de_rand_1_bin,
        MaxTime     = 100.0,
        TraceMode   = :silent
    )

    p̂      = best_candidate(result)
    prob̂   = ODEProblem(model, [y[1]], tspan, p̂)
    x_dense = range(x[1], x[end], length=1000)
    sol̂    = solve(prob̂, solver, reltol=1e-12, abstol=1e-12, saveat=x_dense)

    return p̂, sol̂, prob̂
end

"""
    calculate_bic(prob, xdata, ydata, solver, params)

Calculates the Bayesian Information Criterion (BIC) and Sum of Squared Residuals (SSR) for a solved ODE model.
"""
function calculate_bic(prob, x, y, solver, p)
    sol = solve(prob, solver, reltol=1e-15, abstol=1e-15, saveat=x)
    resid = y .- getindex.(sol.u, 1)
    ssr   = sum(resid .^ 2)
    k     = length(p)
    n     = length(x)
    bic   = n * log(ssr / n) + k * log(n)
    bic, ssr
end

"""
    pQuickStat(x, y, optimized_params, optimized_sol, optimized_prob, bic, ssr)

Displays a plot of model fit and prints model parameters, BIC, and SSR.
"""
function pQuickStat(x, y, p, sol, prob, bic, ssr)
    println("→ Optimized params: ", p)
    println("→ SSR: ", ssr)
    println("→ BIC: ", bic)

    plt = scatter(x, y;
        label   = "Data",
        legend  = :bottomright,
        xlabel  = "Day",
        ylabel  = "Average",
        title   = "Model Fit"
    )
    plot!(plt, sol.t, getindex.(sol.u,1); label="Model", lw=2)
    display(plt)
end



function run_single_fit(
    df::DataFrame,
    p0::Vector{<:Real};
    model         = logistic_growth!,
    fixed_params  = nothing,
    solver        = Rodas5(),
    bounds        = nothing,
    show_stats::Bool = true
)
    # wrap for fixed_params
    if fixed_params !== nothing
        old_model = model
        model = (du,u,p,t) -> old_model(du, u, vcat(p, fixed_params), t)
    end

    nparams = length(p0)
    bounds === nothing && (bounds = [(0.0, Inf) for _ in 1:nparams])

    df_avg = extract_day_averages_from_df(df)
    x      = Float64.(df_avg.Day)
    y      = Float64.(df_avg.Average)
    tspan  = (x[1], x[end])
    u0     = [y[1]]

    p̂, sol̂, prob̂ = setUpProblem(model, x, y, solver, u0, p0, tspan, bounds)
    bic, ssr       = calculate_bic(prob̂, x, y, solver, p̂)
    show_stats && pQuickStat(x, y, p̂, sol̂, prob̂, bic, ssr)

    return (params = p̂, bic = bic, ssr = ssr, sol = sol̂)
end

# ────────────────────────────────────────────────────────────────────────────
# 1. Compare two models on the same dataset
# ────────────────────────────────────────────────────────────────────────────
"""
compare_models(
    df::DataFrame,
    name1::String, model1::Function, p0_1::Vector{<:Real};
    name2::String, model2::Function, p0_2::Vector{<:Real};
    solver               = Rodas5(),
    bounds1              = nothing,
    bounds2              = nothing,
    fixed_params1        = nothing,
    fixed_params2        = nothing,
    show_stats::Bool     = false,
    output_csv::String   = "model_comparison.csv"
)

Fits two candidate models to the same dataset via `run_single_fit`,
plots both curves over the data, prints parameter/BIC/SSR, and writes a CSV summary.
"""
function compare_models(
    df::DataFrame,
    name1::String, model1::Function, p0_1::Vector{<:Real},
    name2::String, model2::Function, p0_2::Vector{<:Real};
    solver             = Rodas5(),
    bounds1            = nothing,
    bounds2            = nothing,
    fixed_params1      = nothing,
    fixed_params2      = nothing,
    show_stats::Bool   = false,
    output_csv::String = "model_comparison.csv"
)
    # Fit model 1
    fit1 = run_single_fit(
        df, p0_1;
        model        = model1,
        fixed_params = fixed_params1,
        solver       = solver,
        bounds       = bounds1,
        show_stats   = show_stats
    )

    # Fit model 2
    fit2 = run_single_fit(
        df, p0_2;
        model        = model2,
        fixed_params = fixed_params2,
        solver       = solver,
        bounds       = bounds2,
        show_stats   = show_stats
    )

    # Extract data for plotting
    df_avg = extract_day_averages_from_df(df)
    x, y   = Float64.(df_avg.Day), Float64.(df_avg.Average)

    # Plot
    plt = scatter(
        x, y;
        label  = "Data",
        xlabel = "Day",
        ylabel = "Value",
        title  = "Model Comparison: $name1 vs $name2",
        legend = :bottomright
    )
    plot!(plt, fit1.sol.t, getindex.(fit1.sol.u,1);
          label=name1, lw=2)
    plot!(plt, fit2.sol.t, getindex.(fit2.sol.u,1);
          label=name2, lw=2, linestyle=:dash)
    display(plt)

    # Print summary
    println("=== $name1 ===")
    println("Params: $(fit1.params), BIC: $(fit1.bic), SSR: $(fit1.ssr)")
    println("=== $name2 ===")
    println("Params: $(fit2.params), BIC: $(fit2.bic), SSR: $(fit2.ssr)")

    # Save CSV
    df_out = DataFrame(
        Model  = [name1, name2],
        Params = [string(fit1.params), string(fit2.params)],
        BIC    = [fit1.bic, fit2.bic],
        SSR    = [fit1.ssr, fit2.ssr]
    )
    CSV.write(output_csv, df_out)
    println("Results saved to $output_csv")
end


# ────────────────────────────────────────────────────────────────────────────
# 2. Compare same or different models across two datasets
# ────────────────────────────────────────────────────────────────────────────
"""
compare_datasets(
    df1::DataFrame, name1::String, model1::Function, p0_1::Vector{<:Real};
    df2::DataFrame, name2::String, model2::Function, p0_2::Vector{<:Real};
    solver               = Rodas5(),
    bounds1              = nothing,
    bounds2              = nothing,
    fixed_params1        = nothing,
    fixed_params2        = nothing,
    show_stats::Bool     = false,
    output_csv::String   = "dataset_comparison.csv"
)

Fits a model to two different datasets via `run_single_fit`,
plots both fits side-by-side, prints stats, and writes a CSV summary.
"""
function compare_datasets(
    df1::DataFrame, name1::String, model1::Function, p0_1::Vector{<:Real},
    df2::DataFrame, name2::String, model2::Function, p0_2::Vector{<:Real};
    solver             = Rodas5(),
    bounds1            = nothing,
    bounds2            = nothing,
    fixed_params1      = nothing,
    fixed_params2      = nothing,
    show_stats::Bool   = false,
    output_csv::String = "dataset_comparison.csv"
)
    # Fit first dataset
    fit1 = run_single_fit(
        df1, p0_1;
        model        = model1,
        fixed_params = fixed_params1,
        solver       = solver,
        bounds       = bounds1,
        show_stats   = show_stats
    )

    # Fit second dataset
    fit2 = run_single_fit(
        df2, p0_2;
        model        = model2,
        fixed_params = fixed_params2,
        solver       = solver,
        bounds       = bounds2,
        show_stats   = show_stats
    )

    # Extract data for plotting
    df_avg1 = extract_day_averages_from_df(df1)
    df_avg2 = extract_day_averages_from_df(df2)
    x1, y1  = Float64.(df_avg1.Day), Float64.(df_avg1.Average)
    x2, y2  = Float64.(df_avg2.Day), Float64.(df_avg2.Average)

    # Plot
    plt = scatter(
        x1, y1;
        label  = "Data - $name1",
        color  = :green,
        xlabel = "Day",
        ylabel = "Value",
        title  = "Dataset Comparison: $name1 vs $name2",
        legend = :bottomright
    )
    plot!(plt, fit1.sol.t, getindex.(fit1.sol.u,1);
          label="Model - $name1", color=:green, lw=2)

    scatter!(plt, x2, y2;
             label  = "Data - $name2",
             color  = :purple)
    plot!(plt, fit2.sol.t, getindex.(fit2.sol.u,1);
          label="Model - $name2", color=:purple, lw=2, linestyle=:dash)
    display(plt)

    # Print summary
    println("=== $name1 ===")
    println("Params: $(fit1.params), BIC: $(fit1.bic), SSR: $(fit1.ssr)")
    println("=== $name2 ===")
    println("Params: $(fit2.params), BIC: $(fit2.bic), SSR: $(fit2.ssr)")

    # Save CSV
    df_out = DataFrame(
        Dataset = [name1, name2],
        Params  = [string(fit1.params), string(fit2.params)],
        BIC     = [fit1.bic, fit2.bic],
        SSR     = [fit1.ssr, fit2.ssr]
    )
    CSV.write(output_csv, df_out)
    println("Results saved to $output_csv")
end

"""
compare_models_dict(
    df::DataFrame,
    specs::Dict{String,<:NamedTuple},;
    default_solver        = Rodas5(),
    show_stats::Bool      = false,
    output_csv::String    = "all_models_comparison.csv"
)

Fits each model in `specs` to `df`, allowing each spec to override solver,
plots all model curves together, prints a summary table, and writes results to CSV.

Each `specs[name]` should be a NamedTuple with fields:
  • model::Function
  • p0::Vector{<:Real}
  • bounds::Vector{Tuple{<:Real,<:Real}}
  • fixed_params::Union{Nothing,Vector{<:Real}}
  • (optional) solver::Any  # e.g. Rodas5() or Tsit5()
"""
function compare_models_dict(
    df::DataFrame,
    specs::Dict{String,<:NamedTuple};
    default_solver        = Rodas5(),
    show_stats::Bool      = false,
    output_csv::String    = "all_models_comparison.csv"
)
    fits = Dict{String,Any}()
    results = NamedTuple[]
    # Fit each model
    for (name, spec) in specs
        solver_i = haskey(spec, :solver) ? spec.solver : default_solver
        fit = run_single_fit(
            df, spec.p0;
            model        = spec.model,
            fixed_params = spec.fixed_params,
            solver       = solver_i,
            bounds       = spec.bounds,
            show_stats   = show_stats
        )
        fits[name] = fit
        push!(results, (
            Model  = name,
            Params = fit.params,
            BIC    = fit.bic,
            SSR    = fit.ssr
        ))
    end

    # Summary DataFrame
    df_summary = DataFrame(
        Model  = [r.Model for r in results],
        Params = [string(r.Params) for r in results],
        BIC    = [r.BIC for r in results],
        SSR    = [r.SSR for r in results]
    )
    # Print BIC table
    println("
BIC Summary:")
    display(df_summary[:, [:Model, :BIC]])

    # Save summary CSV
    CSV.write(output_csv, df_summary)
    println("Summary saved to $output_csv")

    # Plot data + model curves
    df_avg = extract_day_averages_from_df(df)
    x, y = Float64.(df_avg.Day), Float64.(df_avg.Average)
    plt = scatter(x, y;
                  label="Data",
                  xlabel="Day",
                  ylabel="Value",
                  title="All Models Comparison",
                  legend=:bottomright)
    for name in keys(fits)
        fit = fits[name]
        plot!(plt, fit.sol.t, getindex.(fit.sol.u,1);
              label=name, lw=2)
    end
    display(plt)

    # Collect raw predictions
    pred_rows = NamedTuple[]
    for (name, fit) in pairs(fits)
        for (t, u) in zip(fit.sol.t, fit.sol.u)
            push!(pred_rows, (Model=name, Time=t, Prediction=u[1]))
        end
    end
    df_preds = DataFrame(pred_rows)
    preds_csv = replace(output_csv, r"\.csv$" => "_predictions.csv")
    CSV.write(preds_csv, df_preds)
    println("Predictions saved to $preds_csv")

    return fits
end

# 1) plain logistic: p = (r, K)
function logistic_growth!(du,u,p,t)
  r,K = p; du[1] = r*u[1]*(1 - u[1]/K)
end

# 2) logistic + death: p = (r, K, δ)
function logistic_growth_with_death!(du,u,p,t)
  r,K,δ = p; du[1] = r*u[1]*(1 - u[1]/K) - δ*u[1]
end

# 3) Gompertz: p = (a, b)
function gompertz_growth!(du,u,p,t)
  a,b = p; du[1] = a*u[1]*exp(-b*t)
end

# 4) Gompertz + death: p = (a, b, δ)
function gompertz_growth_with_death!(du,u,p,t)
  a,b,δ = p; du[1] = a*u[1]*exp(-b*t) - δ*u[1]

end

# 5) exp with lag: p = (r, t_lag)
function exponential_growth_with_delay!(du,u,p,t)
  r,tlag = p; du[1] = (t>=tlag ? r : 0.0)*u[1]
end

# 6) logistic with lag: p = (r, K, t_lag)
function logistic_growth_with_delay!(du,u,p,t)
  r,K,tlag = p; du[1] = (t>=tlag ? r : 0.0)*u[1]*(1-u[1]/K)
end

end # module V1SimpleODE
