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
    extractData(dfTemp)

Extracts non-missing values from the column "Day Averages" in the given DataFrame and assigns time indices.
Returns two Float64 arrays: time points `x` and values `y`. Note that names for lines need to follow this pattern <celltype>_<drug_concentration>_<treated/untreated>_<Day#>_<Tile-#>_<Well/Sample>.

"""

function extractData(df::DataFrame)
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

"""
    compare_cell_response_models(
        df_res::DataFrame, p0_res::Vector{<:Real};
        df_sen::DataFrame, p0_sen::Vector{<:Real},
        model_res::Function=logistic_growth!,
        model_sen::Function=logistic_growth!,
        solver=Rodas5(),
        bounds_res=nothing,
        bounds_sen=nothing,
        fixed_params_res=nothing,
        fixed_params_sen=nothing,
        show_stats::Bool=false,
        output_csv::String="cell_response_comparison.csv"
    )

Fits two cell-response models (e.g. resistant vs sensitive) using `run_single_fit`,
then plots both fits side by side, prints statistics, and saves a summary CSV.
"""
function compare_cell_response_models(
    df_res::DataFrame, p0_res::Vector{<:Real};
    df_sen::DataFrame, p0_sen::Vector{<:Real},
    model_res::Function=logistic_growth!,
    model_sen::Function=logistic_growth!,
    solver            = Rodas5(),
    bounds_res        = nothing,
    bounds_sen        = nothing,
    fixed_params_res  = nothing,
    fixed_params_sen  = nothing,
    show_stats::Bool  = false,
    output_csv::String="cell_response_comparison.csv"
)
    # Fit resistant
    res = run_single_fit(
        df_res, p0_res;
        model        = model_res,
        fixed_params = fixed_params_res,
        solver       = solver,
        bounds       = bounds_res,
        show_stats   = show_stats
    )

    # Fit sensitive
    sen = run_single_fit(
        df_sen, p0_sen;
        model        = model_sen,
        fixed_params = fixed_params_sen,
        solver       = solver,
        bounds       = bounds_sen,
        show_stats   = show_stats
    )

    # Prepare data for plotting
    df_res_avg = extract_day_averages_from_df(df_res)
    df_sen_avg = extract_day_averages_from_df(df_sen)
    x_res, y_res = Float64.(df_res_avg.Day), Float64.(df_res_avg.Average)
    x_sen, y_sen = Float64.(df_sen_avg.Day), Float64.(df_sen_avg.Average)

    # Plot
    plt = scatter(
        x_res, y_res;
        label  = "Data - Resistant",
        color  = :red,
        xlabel = "Day", ylabel = "Value",
        legend = :bottomright,
        title  = "Resistant vs Sensitive Fit"
    )
    plot!(plt, res.sol.t, getindex.(res.sol.u,1);
          label=:"Model - Resistant", color=:red, lw=2)

    scatter!(plt, x_sen, y_sen; label="Data - Sensitive", color=:blue)
    plot!(plt, sen.sol.t, getindex.(sen.sol.u,1);
          label=:"Model - Sensitive", color=:blue, lw=2, linestyle=:dash)
    display(plt)

    # Print summary
    println("===== Comparison Summary =====")
    println("Resistant -> Params: $(res.params), BIC: $(res.bic), SSR: $(res.ssr)")
    println("Sensitive -> Params: $(sen.params), BIC: $(sen.bic), SSR: $(sen.ssr)")

    # Save CSV
    df_out = DataFrame(
        Label = ["Resistant", "Sensitive"],
        Params = [string(res.params), string(sen.params)],
        BIC    = [res.bic, sen.bic],
        SSR    = [res.ssr, sen.ssr]
    )
    CSV.write(output_csv, df_out)
    println("Results saved to $output_csv")
end

"""
    compare_models(
        df::DataFrame,
        name1::String, model1::Function, p0_1::Vector{<:Real};
        name2::String, model2::Function, p0_2::Vector{<:Real},
        solver=Rodas5(),
        bounds1=nothing, bounds2=nothing,
        fixed_params1=nothing, fixed_params2=nothing,
        show_stats::Bool=false,
        output_csv::String="model_comparison.csv"
    )

Fits two candidate models to the *same* dataset using `run_single_fit`,
plots them together, prints stats, and writes a CSV summary.
"""
function compare_models(
    df::DataFrame,
    name1::String, model1::Function, p0_1::Vector{<:Real};
    name2::String, model2::Function, p0_2::Vector{<:Real},
    solver            = Rodas5(),
    bounds1           = nothing,
    bounds2           = nothing,
    fixed_params1     = nothing,
    fixed_params2     = nothing,
    show_stats::Bool  = false,
    output_csv::String="model_comparison.csv"
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

    # Data for plotting
    df_avg = extract_day_averages_from_df(df)
    x, y   = Float64.(df_avg.Day), Float64.(df_avg.Average)

    plt = scatter(
        x, y; label="Data",
        xlabel="Day", ylabel="Value",
        title="Model Comparison: $name1 vs $name2",
        legend=:bottomright
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

end # module V1SimpleODE
