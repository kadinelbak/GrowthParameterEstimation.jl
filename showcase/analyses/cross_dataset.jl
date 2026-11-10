# Cross-dataset summary, built only from committed results/<id>/ folders (no fitting).
#   julia --project=showcase showcase/analyses/cross_dataset.jl
# Writes results/cross/*.csv and figures/cross/*.{png,svg,pdf}.

isdefined(@__MODULE__, :SHOWCASE) || include(joinpath(@__DIR__, "common.jl"))
isdefined(@__MODULE__, :savefig3) || include(joinpath(@__DIR__, "figures.jl"))

const MANIFEST_PATH = joinpath(SHOWCASE, "data", "manifest.toml")

finished_datasets() = sort([d for d in readdir(RESULTS)
                            if d != "cross" && isfile(joinpath(RESULTS, d, "meta.toml"))])

function cross_tables(ids = finished_datasets())
    man = TOML.parsefile(MANIFEST_PATH)["datasets"]
    crit = DataFrame(dataset = String[], scale = String[], model = String[], k = Int[], n_curves = Int[],
                     delta_bic_per_curve = Float64[], share_picked = Float64[], is_pick = Bool[])
    summ = DataFrame(dataset = String[], title = String[], scale = String[], n_curves = Int[], pick = String[],
                     runner_up = String[], gap_bic = Float64[], ident_share = Float64[],
                     forecast_best = String[], forecast_best_err = Float64[], forecast_pick_err = Float64[],
                     runtime_seconds = Float64[])
    for id in ids
        dir = joinpath(RESULTS, id)
        meta = TOML.parsefile(joinpath(dir, "meta.toml"))
        rk = sort(CSV.read(joinpath(dir, "ranking.csv"), DataFrame), :bic)
        n = meta["n_curves"]
        scale = get(get(man, id, Dict()), "scale", "")
        for r in eachrow(rk)
            push!(crit, (id, scale, r.model, r.k, n, r.delta_bic / n, r.curves_picked / n, r.model == meta["pick"]))
        end
        pick = meta["pick"]
        others = rk.model[rk.model .!= pick]
        gap = rk.bic[findfirst(==(others[1]), rk.model)] - rk.bic[findfirst(==(pick), rk.model)]
        ident = NaN
        pf = joinpath(dir, "profiles.csv")
        if isfile(pf)
            p = CSV.read(pf, DataFrame)
            # share of curves on which every parameter of the pick has a closed interval
            per = combine(groupby(p, :subject), :identifiable => all => :all_ok)
            ident = mean(per.all_ok)
        end
        fb, fbe, fpe = "", NaN, NaN
        ff = joinpath(dir, "forecast_summary.csv")
        if isfile(ff)
            f = sort(CSV.read(ff, DataFrame), :median_rel_error)
            fb, fbe = f.model[1], f.median_rel_error[1]
            j = findfirst(==(pick), f.model)
            j === nothing || (fpe = f.median_rel_error[j])
        end
        push!(summ, (id, get(get(man, id, Dict()), "title", id), scale, n, pick, others[1], gap, ident,
                     fb, fbe, fpe, Float64(get(meta, "runtime_seconds", NaN))))
    end
    return crit, summ
end

function _heatmaps(crit, summ, tag)
    ids = summ.dataset
    models = [m for m in (GROWTH_PANEL ∪ unique(crit.model)) if m in crit.model[in.(crit.dataset, Ref(Set(ids)))]]
    labs = [replace(id, r"(\d{4})_" => s"\1 ") for id in ids]
    M = fill(NaN, length(models), length(ids))
    P = fill(NaN, length(models), length(ids))
    for r in eachrow(crit)
        r.dataset in ids || continue
        i, j = findfirst(==(r.model), models), findfirst(==(r.dataset), ids)
        M[i, j] = min(r.delta_bic_per_curve, 10.0)
        P[i, j] = 100 * r.share_picked
    end
    # grey background = model not in that dataset's panel
    common = (; xticks = (1:length(ids), labs), yticks = (1:length(models), nice.(models)), xrotation = 30,
              yflip = true, tickfontsize = 8, size = (260 + 95 * length(ids), 140 + 34 * length(models)),
              bottom_margin = 10Plots.mm, left_margin = 6Plots.mm, background_color_inside = "#d9d9d9", grid = false)
    h1 = heatmap(1:length(ids), 1:length(models), M; c = cgrad(:Blues, rev = true), clims = (0, 10),
                 colorbar_title = "ΔBIC per curve (capped at 10)", common...)
    for r in eachrow(summ)   # mark the rule pick
        scatter!(h1, [findfirst(==(r.dataset), ids)], [findfirst(==(r.pick), models)];
                 marker = :star5, ms = 9, color = "#e07b1a", msw = 0, label = false)
    end
    h2 = heatmap(1:length(ids), 1:length(models), P; c = :Blues, clims = (0, 100),
                 colorbar_title = "% of curves where the rule picks it", common...)
    return [savefig3(h1, "cross", "delta_bic_heatmap" * tag), savefig3(h2, "cross", "pick_share_heatmap" * tag)]
end

function cross_figures(crit, summ)
    growth = in.(summ.pick, Ref(Set(GROWTH_PANEL)))
    out = String[]
    any(growth) && append!(out, _heatmaps(crit, summ[growth, :], ""))
    any(.!growth) && append!(out, _heatmaps(crit, summ[.!growth, :], "_treated"))
    labs = [replace(id, r"(\d{4})_" => s"\1 ") for id in summ.dataset]
    ok = .!isnan.(summ.ident_share)
    if any(ok)
        s = summ[ok, :]
        b = bar(labs[ok], 100 .* s.ident_share; color = PALETTE.pick, lw = 0, legend = false, ylims = (0, 100),
                ylabel = "% of curves: every parameter\nof the pick has a closed 95% interval",
                xrotation = 30, size = (180 + 90 * nrow(s), 420), grid = false, bottom_margin = 10Plots.mm,
                left_margin = 6Plots.mm, tickfontsize = 8)
        push!(out, savefig3(b, "cross", "identifiability_by_dataset"))
    end
    return out
end

function cross_dataset()
    crit, summ = cross_tables()
    dir = joinpath(RESULTS, "cross")
    mkpath(dir)
    CSV.write(joinpath(dir, "criteria_by_dataset.csv"), crit)
    CSV.write(joinpath(dir, "summary.csv"), summ)
    nrow(summ) > 0 && cross_figures(crit, summ)
    return summ
end

if abspath(PROGRAM_FILE) == @__FILE__
    show(cross_dataset(); allcols = true)
    println()
end
