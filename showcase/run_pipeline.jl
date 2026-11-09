# One open pipeline: tidy CSV in, model comparison + identifiability + forecast + figures out.
#
#   julia --project=showcase -t auto showcase/run_pipeline.jl                      # every dataset in pipeline.toml
#   julia --project=showcase -t auto showcase/run_pipeline.jl my.toml              # your own config
#   julia --project=showcase -t auto showcase/run_pipeline.jl pipeline.toml --only benzekry2014_llc
#
# A dataset is any CSV with one row per measurement and columns for the curve id, time and
# value (names are configurable). Every curve is fitted with every model in the panel; the
# pick is the model with the fewest parameters within `rule_bic` BIC of the best.

include(joinpath(@__DIR__, "analyses", "common.jl"))
include(joinpath(@__DIR__, "analyses", "figures.jl"))

function load_dataset(cfg::Dict)
    path = get(cfg, "csv", joinpath(TIDY, cfg["id"] * ".csv"))
    isabspath(path) || (path = normpath(joinpath(SHOWCASE, path)))
    isfile(path) || error("$(cfg["id"]): $(path) not found (for public datasets run showcase/data/prepare.py first)")
    df = CSV.read(path, DataFrame)
    cols = Dict("subject" => get(cfg, "subject_col", "subject"), "time" => get(cfg, "time_col", "time"),
                "value" => get(cfg, "value_col", "value"), "condition" => get(cfg, "condition_col", "condition"))
    for (std, col) in cols
        std == col && continue
        hasproperty(df, Symbol(col)) || (std == "condition" ? continue : error("$(cfg["id"]): column $(col) missing"))
        rename!(df, Symbol(col) => Symbol(std))
    end
    hasproperty(df, :condition) || (df.condition = fill("all", nrow(df)))
    df.subject = string.(df.subject)
    df.condition = string.(df.condition)
    # optional row filters: [dataset.where] column = value or [values]
    for (col, want) in get(cfg, "where", Dict())
        vals = want isa AbstractVector ? string.(want) : [string(want)]
        df = df[in.(string.(df[!, Symbol(col)]), Ref(Set(vals))), :]
    end
    tmax = get(cfg, "time_max", Inf)
    df = df[.!ismissing.(df.time) .& (coalesce.(df.time, Inf) .<= tmax), :]
    if haskey(cfg, "time_shift_to_zero") && cfg["time_shift_to_zero"]
        df = transform(groupby(df, :subject), :time => (t -> t .- minimum(t)) => :time)
    end
    return df
end

function run_dataset(cfg::Dict, defaults::Dict)
    c = merge(defaults, cfg)
    id = c["id"]
    t0 = time()
    df = load_dataset(c)
    conds = curves(df; min_points = get(c, "min_points", 5))
    max_n = get(c, "max_curves_fitted", 0)
    max_n > 0 && length(conds) > max_n && (conds = conds[1:max_n])
    isempty(conds) && error("$(id): no curve has enough points")
    models = String.(get(c, "models", GROWTH_PANEL))
    @info "$(id): $(length(conds)) curves × $(length(models)) models on $(Threads.nthreads()) threads"

    panel = fit_panel(conds, models; n_starts = get(c, "n_starts", 8))
    summary = summarise_panel(panel)
    out = Dict{Symbol,Any}(:panel => panel, :per_curve => summary.per_curve, :ranking => summary.ranking)
    if get(c, "profiles", true)
        out[:profiles] = profile_all(conds, summary.pick)
    end
    if get(c, "forecast", true)
        fc = forecast_check(conds, models; frac = get(c, "forecast_frac", 0.7))
        if nrow(fc) > 0
            out[:forecast] = fc
            fcs = combine(groupby(fc, :model), :median_rel_error => median => :median_rel_error,
                          :within_20pct => mean => :share_within_20pct)
            out[:forecast_summary] = sort!(fcs, :median_rel_error)
        end
    end
    out[:meta] = Dict("dataset" => id, "script" => "run_pipeline.jl", "pick" => summary.pick,
                      "n_curves" => summary.n_curves, "models" => models,
                      "rule" => "fewest parameters within $(RULE_BIC) BIC of best",
                      "package_version" => string(pkgversion(GrowthParameterEstimation)),
                      "runtime_seconds" => round(time() - t0; digits = 1))
    save_results(id; out...)
    dataset_figures(id; conds, xlabel = get(c, "xlabel", "time"), ylabel = get(c, "ylabel", "size"),
                    max_curves = get(c, "max_curves_plotted", 12), curve_label = get(c, "curve_label", ""))
    @info "$(id): pick $(summary.pick) in $(round(time() - t0; digits = 1)) s"
    return summary
end

function main(args)
    cfg_path = !isempty(args) && !startswith(args[1], "--") ? args[1] : joinpath(@__DIR__, "pipeline.toml")
    k = findfirst(==("--only"), args)
    only = k === nothing ? String[] : args[(k + 1):end]
    cfg = TOML.parsefile(cfg_path)
    defaults = get(cfg, "defaults", Dict{String,Any}())
    for d in cfg["dataset"]
        isempty(only) || d["id"] in only || continue
        get(d, "skip", false) && continue
        run_dataset(d, defaults)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
