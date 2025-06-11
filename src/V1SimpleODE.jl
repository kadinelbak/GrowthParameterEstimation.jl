module V1SimpleODE

using StatsBase
using CSV
using Plots
using DataFrames
using DifferentialEquations
using SciMLSensitivity
using LsqFit
using DifferentialEquations, RecursiveArrayTools, Plots, DiffEqParamEstim
using Optimization, ForwardDiff, OptimizationOptimJL, OptimizationBBO
using BlackBoxOptim

"""
    extract_sample_trajectories(file_name)

    Takes a data file that has two columns. One column of names that are formatted like so:
    "...Replicate2...Day2...Classifier..." and will average all the replicate of each respective day and classifier until you
    ouput a  
"""

function extract_sample_trajectories(file_path::String)
    df = CSV.read(file_path, DataFrame)

    # Rename columns for clarity
    rename!(df, Dict(names(df)[1] => :Sample, names(df)[2] => :Count))

    # === Extract day number from the Sample string ===
    df[!, :Day] = [occursin(r"Day\d+", s) ? parse(Int, match(r"Day(\d+)", s).captures[1]) : missing for s in df.Sample]

    # === Extract base sample name by removing Replicate and Day identifiers ===
    df[!, :Base] = [replace(s, r"_Replicate\d+_Day\d+" => "") for s in df.Sample]

    # Sort for consistency
    sort!(df, [:Base, :Day])

    # === Group and average every blank replicates per Base + Day ===
    grouped = groupby(df, [:Base, :Day])
    summarized = combine(grouped, :Count => mean => :Avg)

    # === Prepare storage and plotting ===
    results = Dict{String, NamedTuple{(:x, :y), Tuple{Vector{Int}, Vector{Float64}}}}()
    plot(title="Sample Group Trajectories", xlabel="Day", ylabel="Avg Cell Count", legend=:topright)

    for base in unique(summarized.Base)
        sub = summarized[summarized.Base .== base, :]
        x = sub.Day
        y = sub.Avg

        # Save result
        results[base] = (x = x, y = y)

        # Log to terminal
        println("\nGroup: $base")
        println("  Days (x): ", x)
        println("  Averages (y): ", y)

        # Add to plot
        plot!(x, y, label=base, lw=2)
    end

    display(current())
    return results
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
function setUpProblem(modelTypeSet, xdataSet, ydataSet, solverSet, u0Set, pSet, tspanSet, boundsSet)
    probSet = ODEProblem(modelTypeSet, u0Set, tspanSet, pSet)
    solSet = solve(probSet, solverSet, saveat=xdataSet, reltol=1e-12, abstol=1e-12)
    cost_functionSet = build_loss_objective(
        probSet, solverSet,
        L2Loss(xdataSet, ydataSet),
        Optimization.AutoForwardDiff();
        maxiters=10000,
        verbose=false
    )
    optsolSet = bboptimize(cost_functionSet; 
        SearchRange = collect(zip([b[1] for b in boundsSet], [b[2] for b in boundsSet])), 
        Method = :de_rand_1_bin, 
        MaxTime = 100.0,
        TraceMode = :silent)

    optimized_paramsSet = best_candidate(optsolSet)
    optimized_probSet = ODEProblem(modelTypeSet, [ydataSet[1]], tspanSet, optimized_paramsSet)
    xdata_denseSet = range(xdataSet[1], xdataSet[end], length=1000)
    optimized_solSet = solve(optimized_probSet, solverSet, reltol=1e-12, abstol=1e-12, saveat=xdata_denseSet)
    return optimized_paramsSet, optimized_solSet, optimized_probSet
end

"""
    calculate_bic(prob, xdata, ydata, solver, params)

Calculates the Bayesian Information Criterion (BIC) and Sum of Squared Residuals (SSR) for a solved ODE model.
"""
function calculate_bic(probbic, xdatabic, ydatabic, solverbic, optparbic)
    solbic = solve(probbic, solverbic, reltol=1e-15, abstol=1e-15, saveat=xdatabic)
    residualsbic = [ydatabic[i] - solbic(xdatabic[i])[1] for i in 1:length(xdatabic)]
    ssrbic = sum(residualsbic .^ 2)
    kbic = length(optparbic)
    nbic = length(xdatabic)
    bic = nbic * log(ssrbic / nbic) + kbic * log(nbic)
    return bic, ssrbic
end

"""
    pQuickStat(x, y, optimized_params, optimized_sol, optimized_prob, bic, ssr)

Displays a plot of model fit and prints model parameters, BIC, and SSR.
"""
function pQuickStat(x, y, optimized_params, optimized_sol, optimized_prob, bic, ssr)
    println("\nOptimized Parameters:")
    println(optimized_params)
    println("\nSum of Squared Residuals (SSR):")
    println(ssr)
    println("\nBayesian Information Criterion (BIC):")
    println(bic)
    p = scatter(x, y, label="Data", legend=:bottomright, title="Model Fit", xlabel="Day", ylabel="Value")
    plot!(optimized_sol.t, [u[1] for u in optimized_sol.u], label="Model", lw=2)
    display(p)
end

"""
    compareCellResponseModels(...)

Compares two models (resistant and sensitive) on different datasets.
Plots both fits, prints stats, and saves results to CSV.
"""
function compareCellResponseModels(
    label_res, x_res, y_res, model_res,
    label_sen, x_sen, y_sen, model_sen,
    solver, u0_res, u0_sen,
    p, tspan, bounds;
    output_csv = "cell_response_comparison.csv"
)
    println("===== Solving for Resistant Cells: $label_res =====")
    opt_params_res, opt_sol_res, opt_prob_res = setUpProblem(model_res, x_res, y_res, solver, u0_res, p, tspan, bounds)
    bic_res, ssr_res = calculate_bic(opt_prob_res, x_res, y_res, solver, opt_params_res)

    println("===== Solving for Sensitive Cells: $label_sen =====")
    opt_params_sen, opt_sol_sen, opt_prob_sen = setUpProblem(model_sen, x_sen, y_sen, solver, u0_sen, p, tspan, bounds)
    bic_sen, ssr_sen = calculate_bic(opt_prob_sen, x_sen, y_sen, solver, opt_params_sen)

    p = plot(title = "Resistant vs Sensitive Model Comparison", xlabel = "Day", ylabel = "Value", legend = :bottomright)
    scatter!(p, x_res, y_res, label = "Data - $label_res", color = :red)
    plot!(p, opt_sol_res.t, [u[1] for u in opt_sol_res.u], label = "Model - $label_res", color = :red, lw = 2)

    scatter!(p, x_sen, y_sen, label = "Data - $label_sen", color = :blue)
    plot!(p, opt_sol_sen.t, [u[1] for u in opt_sol_sen.u], label = "Model - $label_sen", color = :blue, lw = 2, linestyle = :dash)

    display(p)

    println("\n=== Statistical Summary ===")
    println("[$label_res] Params: ", opt_params_res)
    println("[$label_res] BIC: ", bic_res, " | SSR: ", ssr_res)
    println("[$label_sen] Params: ", opt_params_sen)
    println("[$label_sen] BIC: ", bic_sen, " | SSR: ", ssr_sen)
    println("="^60)

    df = DataFrame(
        Label = [label_res, label_sen],
        Model = [string(model_res), string(model_sen)],
        Params = [string(opt_params_res), string(opt_params_sen)],
        BIC = [bic_res, bic_sen],
        SSR = [ssr_res, ssr_sen]
    )
    CSV.write(output_csv, df)
    println("Results saved to $output_csv")
end

"""
    compareModelsBB(name1, name2, model1, model2, xdata, ydata, ...)

Compares two models on the same dataset using BlackBox optimization.
Prints model stats, saves CSV, and plots model fits.
"""
function compareModelsBB(name1, name2, model1, model2, xdata, ydata, solver, u0, p, tspan, bounds; output_csv="model_comparison_results.csv")
    optimized_params1, optimized_sol1, optimized_prob1 = setUpProblem(model1, xdata, ydata, solver, u0, p, tspan, bounds)
    bic1, ssr1 = calculate_bic(optimized_prob1, xdata, ydata, solver, optimized_params1)
    optimized_params2, optimized_sol2, optimized_prob2 = setUpProblem(model2, xdata, ydata, solver, u0, p, tspan, bounds)
    bic2, ssr2 = calculate_bic(optimized_prob2, xdata, ydata, solver, optimized_params2)

    println("=== " * name1 * " ===")
    println("Optimized Params: ", optimized_params1)
    println("BIC: ", bic1)
    println("SSR: ", ssr1)
    println("\n=== " * name2 * " ===")
    println("Optimized Params: ", optimized_params2)
    println("BIC: ", bic2)
    println("SSR: ", ssr2)

    results = DataFrame(
        Model = [name1, name2],
        Params = [string(optimized_params1), string(optimized_params2)],
        BIC = [bic1, bic2],
        SSR = [ssr1, ssr2]
    )
    CSV.write(output_csv, results)
    println("Results saved to: $output_csv")

    p1 = scatter(xdata, ydata, label="Observed", xlabel="Time", ylabel="Value", title="Model Fit Comparison")
    plot!(p1, optimized_sol1, label=name1, linewidth=2)
    plot!(p1, optimized_sol2, label=name2, linewidth=2, linestyle=:dash)
    scatter(xdata, ydata, label="Data", legend=:bottomright, title="Model Fit", xlabel="Day", ylabel="Value")
    plot!(p1, layout=(2, 1), size=(800, 600))
end

end
