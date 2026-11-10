# Figures for one showcase dataset, drawn from committed results plus the tidy data.
# Every figure is written as PNG, SVG and PDF under showcase/figures/<id>/.

isdefined(@__MODULE__, :SHOWCASE) || include(joinpath(@__DIR__, "common.jl"))
using Plots

const FIGURES = joinpath(SHOWCASE, "figures")
const PALETTE = (pick = "#1f5fa8", second = "#e07b1a", data = "#222222", grey = "#9a9a9a")

nice(m) = replace(replace(m, "_growth" => ""), "_" => " ")

function savefig3(p, id, name)
    dir = joinpath(FIGURES, id)
    mkpath(dir)
    for ext in ("png", "svg", "pdf")
        savefig(p, joinpath(dir, "$(name).$(ext)"))
    end
    return joinpath(dir, "$(name).png")
end

parse_params(s) = parse.(Float64, split(string(s), ";"))

"""Small multiples: data with the dataset-level pick and runner-up on shared axes."""
function fit_gallery(id, conds, panel, ranking; max_curves = 12, xlabel = "time", ylabel = "size", curve_label = "")
    pick, second = ranking.model[1:2]
    pick_rule = TOML.parsefile(joinpath(RESULTS, id, "meta.toml"))["pick"]
    models = unique([pick_rule, pick == pick_rule ? second : pick])
    shown = conds[1:min(max_curves, length(conds))]
    xmax = maximum(maximum(c.time) for c in shown)
    ymax = 1.05 * maximum(maximum(c.count) for c in shown)
    plots = Any[]
    for c in shown
        p = plot(; title = curve_label * c.name, titlefontsize = 7, xlims = (0, xmax), ylims = (0, ymax),
                 tickfontsize = 6, legend = false, grid = false)
        tt = range(c.time[1], c.time[end], length = 200)
        for (j, m) in enumerate(models)
            row = panel[(string.(panel.subject) .== c.name) .& (panel.model .== m), :]
            (nrow(row) == 0 || row.params[1] isa Missing || isempty(string(row.params[1]))) && continue
            sim = simulate(get_model(m), collect(tt), parse_params(row.params[1]); u0 = c.u0, exposure = c.exposure)
            sim.success && plot!(p, tt, sim.observed; color = j == 1 ? PALETTE.pick : PALETTE.second,
                                 lw = 1.6, ls = j == 1 ? :solid : :dash)
        end
        scatter!(p, c.time, c.count; color = PALETTE.data, ms = 2.5, msw = 0)
        push!(plots, p)
    end
    n = length(plots)
    cols = min(4, n)
    rows = cld(n, cols)
    legend = plot(; framestyle = :none, legend = :top, legend_column = 3, legendfontsize = 8)
    plot!(legend, [NaN], [NaN]; color = PALETTE.pick, lw = 2, label = "$(nice(models[1])) (rule pick)")
    length(models) > 1 && plot!(legend, [NaN], [NaN]; color = PALETTE.second, lw = 2, ls = :dash, label = nice(models[2]))
    scatter!(legend, [NaN], [NaN]; color = PALETTE.data, msw = 0, label = "data")
    fig = plot(plots..., legend; layout = @layout([grid(rows, cols); l{0.04h}]),
               size = (230 * cols, 180 * rows + 40), xlabel = "", ylabel = "",
               left_margin = 3Plots.mm, bottom_margin = 2Plots.mm)
    plot!(fig; plot_title = "$(ylabel) vs $(xlabel)", plot_titlefontsize = 9)
    return savefig3(fig, id, "fit_gallery")
end

"""Dataset-level comparison: summed BIC, AIC and AICc relative to the best model."""
function criteria_bars(id, ranking)
    r = sort(ranking, :bic)
    labels = nice.(r.model)
    d(x) = x .- minimum(x)
    vals = (("ΔBIC", d(r.bic), PALETTE.pick), ("ΔAIC", d(r.aic), PALETTE.second), ("ΔAICc", d(r.aicc), PALETTE.grey))
    x = collect(1:length(labels))
    top = maximum(maximum(v[2]) for v in vals)
    fig = plot(; ylabel = "Δ from best (sum over curves)", xticks = (x, labels), xrotation = 30, ylims = (0, 1.08 * top),
               legend = :outerbottom, legend_column = 3, size = (820, 480), grid = false,
               bottom_margin = 8Plots.mm, left_margin = 5Plots.mm, tickfontsize = 8)
    for (j, (lab, v, col)) in enumerate(vals)
        bar!(fig, x .+ (j - 2) * 0.27, v; bar_width = 0.27, color = col, lw = 0, label = lab)
    end
    return savefig3(fig, id, "criteria")
end

"""How often each model is the per-curve pick under the 2-BIC rule."""
function pick_counts(id, ranking)
    r = sort(ranking, :curves_picked; rev = true)
    fig = bar(nice.(r.model), r.curves_picked; color = PALETTE.pick, lw = 0, legend = false,
              ylabel = "curves where the rule picks it", xrotation = 30, size = (820, 420), grid = false,
              bottom_margin = 8Plots.mm, left_margin = 5Plots.mm, tickfontsize = 8)
    return savefig3(fig, id, "pick_counts")
end

"""Share of curves whose profile interval is closed on both sides, per parameter."""
function identifiability_bars(id, profiles)
    s = combine(groupby(profiles, :parameter), :identifiable => mean => :share)
    fig = bar(s.parameter, 100 .* s.share; color = PALETTE.pick, lw = 0, legend = false, ylims = (0, 100),
              ylabel = "% of curves with a closed 95% interval", size = (520, 380), grid = false,
              left_margin = 5Plots.mm)
    return savefig3(fig, id, "identifiability")
end

function forecast_bars(id, fc)
    r = sort(fc, :median_rel_error)
    fig = bar(nice.(r.model), 100 .* r.median_rel_error; color = PALETTE.second, lw = 0, legend = false,
              ylabel = "median forecast error on held-out late points (%)", xrotation = 30,
              size = (820, 420), grid = false, bottom_margin = 8Plots.mm, left_margin = 5Plots.mm, tickfontsize = 8)
    return savefig3(fig, id, "forecast")
end

function dataset_figures(id; conds, xlabel = "time", ylabel = "size", max_curves = 12, curve_label = "")
    dir = joinpath(RESULTS, id)
    rd(n) = CSV.read(joinpath(dir, "$(n).csv"), DataFrame)
    panel, ranking = rd("panel"), rd("ranking")
    out = [fit_gallery(id, conds, panel, ranking; max_curves, xlabel, ylabel, curve_label), criteria_bars(id, ranking),
           pick_counts(id, ranking)]
    isfile(joinpath(dir, "profiles.csv")) && push!(out, identifiability_bars(id, rd("profiles")))
    isfile(joinpath(dir, "forecast_summary.csv")) && push!(out, forecast_bars(id, rd("forecast_summary")))
    return out
end
