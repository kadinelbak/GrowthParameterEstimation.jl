module Workflow

using CSV
using DataFrames
using Dates
using Random
using Statistics
using TOML
using Optimization
using OptimizationOptimJL

using ..DataLayer
using ..Exposure
using ..Registry
using ..Simulation

export FitCondition, PipelineConfig,
       default_config, save_config, load_config,
       build_conditions, fit, rank_models, plot_topk, export_results, run_pipeline

struct FitCondition
    name::String
    time::Vector{Float64}
    count::Vector{Float64}
    error::Vector{Float64}
    u0::Vector{Float64}
    exposure::Exposure.AbstractExposure
    metadata::Dict{Symbol,Any}
end

struct PipelineConfig
    version::String
    model_names::Vector{String}
    n_starts::Int
    top_k::Int
    maxiters::Int
    reltol::Float64
    abstol::Float64
    weighted::Bool
    seed::Int
    output_dir::String
end

function default_config(; output_dir::String = "results")
    return PipelineConfig(
        "1.0.0",
        Registry.list_models(),
        20,
        5,
        800,
        1e-8,
        1e-8,
        true,
        42,
        output_dir,
    )
end

function save_config(path::AbstractString, cfg::PipelineConfig)
    dict = Dict(
        "version" => cfg.version,
        "model_names" => cfg.model_names,
        "n_starts" => cfg.n_starts,
        "top_k" => cfg.top_k,
        "maxiters" => cfg.maxiters,
        "reltol" => cfg.reltol,
        "abstol" => cfg.abstol,
        "weighted" => cfg.weighted,
        "seed" => cfg.seed,
        "output_dir" => cfg.output_dir,
    )
    open(path, "w") do io
        TOML.print(io, dict)
    end
    return path
end

function load_config(path::AbstractString)
    cfg = TOML.parsefile(path)
    return PipelineConfig(
        string(get(cfg, "version", "1.0.0")),
        String.(get(cfg, "model_names", Registry.list_models())),
        Int(get(cfg, "n_starts", 20)),
        Int(get(cfg, "top_k", 5)),
        Int(get(cfg, "maxiters", 800)),
        Float64(get(cfg, "reltol", 1e-8)),
        Float64(get(cfg, "abstol", 1e-8)),
        Bool(get(cfg, "weighted", true)),
        Int(get(cfg, "seed", 42)),
        string(get(cfg, "output_dir", "results")),
    )
end

function build_conditions(df::DataFrame; condition_cols::Vector{Symbol} = [:dose, :cell_line, :density, :replicate])
    DataLayer.validate_timeseries(df)
    grouped = groupby(df, condition_cols)

    conditions = FitCondition[]
    for g in grouped
        g_sorted = sort(g, :time)
        metadata = Dict{Symbol,Any}(c => g_sorted[1, c] for c in condition_cols)
        cname = join(["$(c)=$(metadata[c])" for c in condition_cols], " | ")

        dose_value = haskey(metadata, :dose) ? Float64(metadata[:dose]) : 0.0
        exposure = Exposure.ConstantExposure(dose_value)

        push!(conditions, FitCondition(
            cname,
            Float64.(g_sorted.time),
            Float64.(g_sorted.count),
            Float64.(g_sorted.error),
            [max(1e-12, Float64(g_sorted.count[1]))],
            exposure,
            metadata,
        ))
    end
    return conditions
end

function _canonical_param(param::Symbol, ties::Dict{Symbol,Symbol})
    seen = Set{Symbol}()
    cur = param
    while haskey(ties, cur) && !(cur in seen)
        push!(seen, cur)
        cur = ties[cur]
    end
    return cur
end

function _build_layout(
    spec::Registry.ModelSpec,
    n_conditions::Int;
    shared_params::Vector{Symbol},
    fixed_params::Dict{Symbol,Float64},
    tie_constraints::Dict{Symbol,Symbol},
)
    pnames = spec.param_names

    shared_set = Set(shared_params)
    condition_specific = Set(setdiff(pnames, shared_params))

    bounds_map = Dict(name => spec.bounds[i] for (i, name) in enumerate(pnames))

    index_map = Dict{Tuple{Symbol,Int},Int}()
    theta_lb = Float64[]
    theta_ub = Float64[]
    theta_name = Tuple{Symbol,Int}[]

    next_idx = 1

    for name in pnames
        if haskey(fixed_params, name)
            continue
        end
        if name in shared_set
            canon = _canonical_param(name, tie_constraints)
            key = (canon, 0)
            if !haskey(index_map, key)
                index_map[key] = next_idx
                b = bounds_map[canon]
                push!(theta_lb, b[1]); push!(theta_ub, b[2])
                push!(theta_name, key)
                next_idx += 1
            end
        end
    end

    for cond_idx in 1:n_conditions
        for name in pnames
            if haskey(fixed_params, name)
                continue
            end
            if name in condition_specific
                canon = _canonical_param(name, tie_constraints)
                key = (canon, cond_idx)
                if !haskey(index_map, key)
                    index_map[key] = next_idx
                    b = bounds_map[canon]
                    push!(theta_lb, b[1]); push!(theta_ub, b[2])
                    push!(theta_name, key)
                    next_idx += 1
                end
            end
        end
    end

    return index_map, theta_lb, theta_ub, shared_set, condition_specific
end

function _expand_params(
    theta::Vector{Float64},
    spec::Registry.ModelSpec,
    cond_idx::Int,
    index_map::Dict{Tuple{Symbol,Int},Int},
    shared_set::Set{Symbol},
    fixed_params::Dict{Symbol,Float64},
    tie_constraints::Dict{Symbol,Symbol},
)
    p = zeros(Float64, length(spec.param_names))
    for (i, name) in enumerate(spec.param_names)
        if haskey(fixed_params, name)
            p[i] = fixed_params[name]
            continue
        end
        canon = _canonical_param(name, tie_constraints)
        if name in shared_set
            p[i] = theta[index_map[(canon, 0)]]
        else
            p[i] = theta[index_map[(canon, cond_idx)]]
        end
    end
    return p
end

function _initial_theta(index_map, lb::Vector{Float64}, ub::Vector{Float64}, rng::AbstractRNG)
    theta = similar(lb)
    for i in eachindex(lb)
        width = max(ub[i] - lb[i], 1e-12)
        theta[i] = lb[i] + rand(rng) * width
    end
    return theta
end

function fit(
    spec::Registry.ModelSpec,
    conditions::Vector{FitCondition};
    shared_params::Vector{Symbol} = copy(spec.param_names),
    fixed_params::Dict{Symbol,Float64} = Dict{Symbol,Float64}(),
    tie_constraints::Dict{Symbol,Symbol} = Dict{Symbol,Symbol}(),
    n_starts::Int = 20,
    maxiters::Int = 800,
    weighted::Bool = true,
    reltol::Float64 = 1e-8,
    abstol::Float64 = 1e-8,
    seed::Int = 42,
    top_k::Int = 5,
)
    rng = MersenneTwister(seed)
    n_conditions = length(conditions)
    n_obs = sum(length(c.time) for c in conditions)

    index_map, lb, ub, shared_set, condition_specific = _build_layout(
        spec,
        n_conditions;
        shared_params=shared_params,
        fixed_params=fixed_params,
        tie_constraints=tie_constraints,
    )

    if isempty(lb)
        error("No free parameters remain after fixed/tie constraints")
    end

    failure_log = DataFrame(
        stage=String[],
        model=String[],
        condition=String[],
        reason=String[],
        timestamp=String[],
    )

    function objective(theta, _)
        sse = 0.0
        penalty = 0.0

        for (ci, cond) in enumerate(conditions)
            p = _expand_params(theta, spec, ci, index_map, shared_set, fixed_params, tie_constraints)
            sim = Simulation.simulate(
                spec,
                cond.time,
                p;
                u0=cond.u0,
                exposure=cond.exposure,
                reltol=reltol,
                abstol=abstol,
            )

            if !sim.success
                penalty += 1e12
                continue
            end

            resid = cond.count .- sim.observed
            if weighted
                w = 1.0 ./ max.(cond.error .^ 2, 1e-12)
                sse += sum(w .* (resid .^ 2))
            else
                sse += sum(resid .^ 2)
            end
        end

        return sse + penalty
    end

    loss = OptimizationFunction((x, p) -> objective(x, p), Optimization.AutoForwardDiff())

    rows = NamedTuple[]
    best = nothing

    for start_id in 1:n_starts
        theta0 = if start_id == 1
            (lb .+ ub) ./ 2
        else
            _initial_theta(index_map, lb, ub, rng)
        end

        try
            prob = OptimizationProblem(loss, theta0; lb=lb, ub=ub)
            result = solve(prob, OptimizationOptimJL.Fminbox(OptimizationOptimJL.BFGS()); maxiters=maxiters)

            theta_hat = collect(result.u)
            obj = objective(theta_hat, nothing)

            k = length(theta_hat)
            aic = n_obs * log(max(obj, 1e-12) / n_obs) + 2k
            bic = n_obs * log(max(obj, 1e-12) / n_obs) + k * log(n_obs)

            row = (
                start_id=start_id,
                objective=obj,
                converged=result.retcode == ReturnCode.Success,
                retcode=string(result.retcode),
                aic=aic,
                bic=bic,
                params=theta_hat,
            )
            push!(rows, row)

            if isnothing(best) || obj < best.objective
                best = row
            end
        catch err
            push!(failure_log, (
                stage="optimization",
                model=spec.name,
                condition="all",
                reason=Simulation.classify_failure(err),
                timestamp=string(now()),
            ))
        end
    end

    if isempty(rows)
        error("All optimization starts failed for model $(spec.name)")
    end

    sorted_rows = sort(rows, by=r -> r.objective)
    keep = sorted_rows[1:min(top_k, length(sorted_rows))]

    per_condition = NamedTuple[]
    for (ci, cond) in enumerate(conditions)
        pbest = _expand_params(best.params, spec, ci, index_map, shared_set, fixed_params, tie_constraints)
        sim = Simulation.simulate(
            spec,
            cond.time,
            pbest;
            u0=cond.u0,
            exposure=cond.exposure,
            reltol=reltol,
            abstol=abstol,
        )
        if !sim.success
            push!(failure_log, (
                stage="solver",
                model=spec.name,
                condition=cond.name,
                reason=sim.reason,
                timestamp=string(now()),
            ))
        end

        push!(per_condition, (
            condition=cond.name,
            params=pbest,
            observed=sim.observed,
            residuals=sim.success ? (cond.count .- sim.observed) : fill(NaN, length(cond.time)),
            success=sim.success,
            reason=sim.reason,
        ))
    end

    return (
        model=spec.name,
        best=best,
        top_fits=keep,
        n_obs=n_obs,
        n_params=length(best.params),
        per_condition=per_condition,
        failures=failure_log,
    )
end

function rank_models(
    model_names::Vector{String},
    conditions::Vector{FitCondition};
    top_k::Int = 5,
    kwargs...
)
    rows = NamedTuple[]
    fit_map = Dict{String,Any}()
    all_failures = DataFrame(stage=String[], model=String[], condition=String[], reason=String[], timestamp=String[])

    for name in model_names
        spec = Registry.get_model(name)
        try
            fres = fit(spec, conditions; top_k=top_k, kwargs...)
            fit_map[name] = fres
            push!(rows, (
                model=name,
                sse=fres.best.objective,
                weighted_sse=fres.best.objective,
                aic=fres.best.aic,
                bic=fres.best.bic,
                n_params=fres.n_params,
            ))
            append!(all_failures, fres.failures)
        catch err
            push!(rows, (
                model=name,
                sse=Inf,
                weighted_sse=Inf,
                aic=Inf,
                bic=Inf,
                n_params=0,
            ))
            push!(all_failures, (
                stage="model_fit",
                model=name,
                condition="all",
                reason=Simulation.classify_failure(err),
                timestamp=string(now()),
            ))
        end
    end

    ranking = DataFrame(rows)
    sort!(ranking, :bic)
    best_bic = minimum(ranking.bic)
    ranking.delta_bic = ranking.bic .- best_bic

    return (ranking=ranking, fits=fit_map, failures=all_failures)
end

function _load_plots_or_nothing()
    try
        @eval using Plots
        return true
    catch
        return false
    end
end

function plot_topk(
    rank_result;
    conditions::Vector{FitCondition},
    top_k::Int = 5,
    output_dir::String = "results/figures",
)
    mkpath(output_dir)
    ranking = rank_result.ranking
    top_models = ranking.model[1:min(top_k, nrow(ranking))]

    has_plots = _load_plots_or_nothing()
    generated = String[]

    for cond in conditions
        overlay = DataFrame(time=cond.time, observed=cond.count)
        for m in top_models
            fit_info = rank_result.fits[m]
            cond_hit = findfirst(pc -> pc.condition == cond.name, fit_info.per_condition)
            if !isnothing(cond_hit)
                overlay[!, Symbol("pred_" * m)] = fit_info.per_condition[cond_hit].observed
            end
        end

        csv_path = joinpath(output_dir, replace(cond.name, '|' => '_', ' ' => '_') * "_overlay.csv")
        CSV.write(csv_path, overlay)
        push!(generated, csv_path)

        if has_plots
            p = Plots.plot(cond.time, cond.count; seriestype=:scatter, label="data", title=cond.name, xlabel="time", ylabel="count")
            for m in top_models
                col = Symbol("pred_" * m)
                if col in names(overlay)
                    Plots.plot!(p, overlay.time, overlay[!, col]; label=m)
                end
            end
            png_path = replace(csv_path, ".csv" => ".png")
            Plots.savefig(p, png_path)
            push!(generated, png_path)
        end
    end

    return generated
end

function export_results(
    rank_result;
    output_dir::String = "results",
)
    tables_dir = joinpath(output_dir, "tables")
    params_dir = joinpath(output_dir, "params")
    diag_dir = joinpath(output_dir, "diagnostics")
    fig_dir = joinpath(output_dir, "figures")

    for d in (tables_dir, params_dir, diag_dir, fig_dir)
        mkpath(d)
    end

    ranking_path = joinpath(tables_dir, "model_ranking.csv")
    CSV.write(ranking_path, rank_result.ranking)

    best_model = rank_result.ranking.model[1]
    best_fit = rank_result.fits[best_model]

    starts_df = DataFrame(best_fit.top_fits)
    starts_path = joinpath(tables_dir, "top_fit_starts.csv")
    CSV.write(starts_path, starts_df)

    p_df = DataFrame(param_index=collect(1:length(best_fit.best.params)), value=best_fit.best.params)
    params_path = joinpath(params_dir, "best_params.csv")
    CSV.write(params_path, p_df)

    fail_path = joinpath(diag_dir, "failure_report.csv")
    CSV.write(fail_path, rank_result.failures)

    summary = DataFrame(
        key=["best_model", "best_bic", "generated_at"],
        value=[best_model, string(rank_result.ranking.bic[1]), string(now())],
    )
    summary_path = joinpath(output_dir, "best_model_summary.csv")
    CSV.write(summary_path, summary)

    return (
        ranking=ranking_path,
        starts=starts_path,
        params=params_path,
        failures=fail_path,
        summary=summary_path,
        figures=fig_dir,
    )
end

function run_pipeline(
    data_input;
    config::PipelineConfig = default_config(),
    include_models::Vector{String} = String[],
    exclude_models::Vector{String} = String[],
)
    df = if data_input isa DataFrame
        DataLayer.normalize_schema(data_input)
    elseif data_input isa AbstractString
        DataLayer.load_timeseries(data_input)
    else
        error("data_input must be a DataFrame or file path")
    end

    DataLayer.validate_timeseries(df)
    conditions = build_conditions(df)

    model_names = isempty(include_models) ? copy(config.model_names) : include_models
    if !isempty(exclude_models)
        model_names = setdiff(model_names, exclude_models)
    end

    rank_result = rank_models(
        model_names,
        conditions;
        top_k=config.top_k,
        n_starts=config.n_starts,
        maxiters=config.maxiters,
        weighted=config.weighted,
        reltol=config.reltol,
        abstol=config.abstol,
        seed=config.seed,
    )

    plot_paths = plot_topk(rank_result; conditions=conditions, top_k=config.top_k, output_dir=joinpath(config.output_dir, "figures"))
    export_paths = export_results(rank_result; output_dir=config.output_dir)

    return (
        config=config,
        conditions=conditions,
        ranking=rank_result.ranking,
        failures=rank_result.failures,
        plots=plot_paths,
        exports=export_paths,
    )
end

end # module Workflow
