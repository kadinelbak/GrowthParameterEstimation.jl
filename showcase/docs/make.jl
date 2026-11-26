# Build the showcase site from committed results (no fitting happens here).
#   julia --project=showcase showcase/docs/make.jl
using Documenter, CSV, DataFrames, TOML, Printf

const SHOWCASE = normpath(joinpath(@__DIR__, ".."))
const SRC = joinpath(@__DIR__, "src")
const GEN = joinpath(SRC, "datasets")
const MANIFEST = TOML.parsefile(joinpath(SHOWCASE, "data", "manifest.toml"))["datasets"]
# Agreement of each page with its source paper, written by hand in docs/verdicts.toml.
const VERDICTS = TOML.parsefile(joinpath(@__DIR__, "verdicts.toml"))
const VERDICT_STYLE = Dict(
    "match" => ("tip", "✅ Matches the paper"),
    "partial" => ("warning", "⚠️ Partly matches the paper"),
    "differs" => ("danger", "❌ Does not match the paper"),
    "not_comparable" => ("note", "➖ Not directly comparable with the paper"),
)
verdict_label(id) = haskey(VERDICTS, id) ? VERDICT_STYLE[VERDICTS[id]["verdict"]][2] : "—"

"""A coloured box at the top of a dataset page: verdict, the paper's finding, ours, and why."""
function verdict_block(id)
    haskey(VERDICTS, id) || return ""
    v = VERDICTS[id]
    kind, label = VERDICT_STYLE[v["verdict"]]
    io = IOBuffer()
    println(io, "!!! ", kind, " \"", label, "\"")
    println(io, "    **The paper found:** ", v["paper"], "\n")
    println(io, "    **This page found:** ", v["ours"], "\n")
    println(io, "    **Why they differ / caveats:** ", v["why"], "\n")
    return String(take!(io))
end

nice(m) = replace(replace(m, "_growth" => ""), "_" => " ")
fmt(x::Real) = isfinite(x) ? @sprintf("%.1f", x) : "—"
fmtp(x::Real) = !isfinite(x) ? "—" : (abs(x) >= 1e4 || abs(x) < 1e-2) ? @sprintf("%.3g", x) : @sprintf("%.3f", x)

function md_table(header, rows)
    io = IOBuffer()
    println(io, "| ", join(header, " | "), " |")
    println(io, "|", join(fill(" :--- ", length(header)), "|"), "|")
    for r in rows
        println(io, "| ", join(r, " | "), " |")
    end
    return String(take!(io))
end

function dataset_page(id)
    res = joinpath(SHOWCASE, "results", id)
    meta = TOML.parsefile(joinpath(res, "meta.toml"))
    src = MANIFEST[get(meta, "source", id)]
    rd(n) = CSV.read(joinpath(res, "$(n).csv"), DataFrame)
    ranking = rd("ranking")
    io = IOBuffer()
    println(io, "# ", get(meta, "title", src["title"]), "\n")
    println(io, verdict_block(id))
    println(io, "*", src["citation"], "* · [doi:", src["doi"], "](https://doi.org/", src["doi"], ") · source: [", src["source"], "](", src["source"], ") · licence: ", src["licence"], "\n")
    intro = joinpath(@__DIR__, "intro", "$(id).md")
    isfile(intro) && println(io, read(intro, String), "\n")

    println(io, "## Model comparison\n")
    println(io, "Each of the ", meta["n_curves"], " curves was fitted separately with every model. ",
            "Criteria are summed over curves and shown as differences from the best model. ",
            "The rule picks the model with the fewest parameters within 2 BIC of the best: **", nice(meta["pick"]), "**.\n")
    rows = [[nice(r.model) * (r.model == meta["pick"] ? " **(pick)**" : ""), string(r.k), fmt(r.delta_bic),
             fmt(r.delta_aic), fmt(r.aicc - minimum(ranking.aicc)), string(r.curves_picked)] for r in eachrow(ranking)]
    println(io, md_table(["Model", "k", "ΔBIC", "ΔAIC", "ΔAICc", "Curves where picked"], rows))
    println(io, "\n![Fits](../assets/", id, "/fit_gallery.png)\n")
    println(io, "![Criteria](../assets/", id, "/criteria.png)\n")
    println(io, "![Per-curve picks](../assets/", id, "/pick_counts.png)\n")

    if isfile(joinpath(res, "profiles.csv"))
        prof = rd("profiles")
        println(io, "## Identifiability of the pick\n")
        println(io, "Profile likelihood on every curve (each parameter fixed on a grid, the others refitted; 95% cutoff χ²₁ = 3.84). ",
                "A parameter counts as identifiable on a curve when its interval closes inside the bounds.\n")
        s = combine(groupby(prof, :parameter), :identifiable => (x -> round(100 * sum(x) / length(x))) => :pct,
                    :estimate => (x -> median(x)) => :med)
        println(io, md_table(["Parameter", "Median estimate", "Curves identifiable (%)"],
                             [[r.parameter, fmtp(r.med), string(Int(r.pct))] for r in eachrow(s)]))
        println(io, "\n![Identifiability](../assets/", id, "/identifiability.png)\n")
    end
    if isfile(joinpath(res, "forecast_summary.csv"))
        fc = rd("forecast_summary")
        println(io, "## Forecast check\n")
        println(io, "Each model was refitted on the first 70% of each curve and asked to predict the rest.\n")
        println(io, md_table(["Model", "Median relative error (%)", "Predictions within 20% (%)"],
                             [[nice(r.model), fmt(100 * r.median_rel_error), fmt(100 * r.share_within_20pct)] for r in eachrow(fc)]))
        println(io, "\n![Forecast](../assets/", id, "/forecast.png)\n")
    end
    isfile(joinpath(res, "checks_fisher.csv")) && package_checks_section(io, id, res, rd)
    println(io, "## Reproduce\n")
    println(io, "```bash\npython3 -I showcase/data/prepare.py --only ", get(meta, "source", id), "\njulia --project=showcase -t auto showcase/run_pipeline.jl --only ", id, "\n```\n")
    println(io, "Fitted with GrowthParameterEstimation.jl ", meta["package_version"], ". Results: `showcase/results/", id, "/`.")
    path = joinpath(GEN, "$(id).md")
    write(path, String(take!(io)))
    copy_figs(id)   # figures next to the page
    return "datasets/$(id).md"
end

relwidth(lo, hi, est) = all(isfinite, (lo, hi, est)) && est != 0 ? 100 * (hi - lo) / abs(est) : NaN
medfin(x) = (v = filter(isfinite, collect(skipmissing(x))); isempty(v) ? NaN : median(v))
pct(x) = isempty(x) ? "—" : string(round(Int, 100 * count(x) / length(x)))

# closed = a grid point below and above the interval failed the cutoff, and the interval has
# width. Checked here because the package also labels as `bounded` an interval that reaches
# the edge of its grid (clipped grids, or a bound recomputed with rounding error).
inside(v, edge) = abs(v - edge) > 1e-6 * max(abs(edge), 1e-12)
closed(d) = (d.status .== "bounded") .& (d.upper .> d.lower) .& inside.(d.lower, d.grid_min) .& inside.(d.upper, d.grid_max)

"""Read the package-diagnostics tables (written by analyses/package_checks.jl) for one dataset."""
function read_checks(res)
    rd(n) = (d = CSV.read(joinpath(res, "$(n).csv"), DataFrame); d.subject = string.(d.subject); d.model = String.(d.model); d)
    pg = rd("checks_profile")
    # use the cutoff crossing interpolated between grid points where there is one
    pg.lower = ifelse.(isfinite.(pg.lower_interp), pg.lower_interp, pg.lower)
    pg.upper = ifelse.(isfinite.(pg.upper_interp), pg.upper_interp, pg.upper)
    return (fi = rd("checks_fisher"), pg, bs = rd("checks_boot"), rc = rd("checks_recovery"), cv = rd("checks_cv"))
end

"""One model's diagnostics, with the Fisher-vs-profile agreement count."""
function checks_data(all, model)
    f(d) = d[d.model .== model, :]
    fi, pg, bs, rc, cv = f(all.fi), f(all.pg), f(all.bs), f(all.rc), f(all.cv)
    pg2 = copy(pg)
    pg2.closed = closed(pg)
    j = innerjoin(select(fi, :subject, :parameter, [:wald_lower, :wald_upper, :estimate] => ByRow(relwidth) => :wf),
                  select(pg2, :subject, :parameter, :closed, [:lower, :upper, :estimate] => ByRow(relwidth) => :wp),
                  on = [:subject, :parameter])
    ok = j[j.closed .& isfinite.(j.wf) .& isfinite.(j.wp), :]
    agree = isempty(ok) ? 0 : count(r -> 2 / 3 <= r.wf / r.wp <= 1.5, eachrow(ok))
    return (; fi, pg, bs, rc, cv, n = length(unique(fi.subject)), agree, pairs = nrow(j))
end

share_str(a, b) = b == 0 ? "—" : string(round(Int, 100 * a / b))

"""The package's own diagnostics on a subset of curves, for every model, as a dataset-page section."""
function package_checks_section(io, id, res, rd)
    all = read_checks(res)
    pick = TOML.parsefile(joinpath(res, "meta.toml"))["pick"]
    models = unique(all.fi.model)
    d = checks_data(all, pick)
    own = isfile(joinpath(res, "profiles.csv")) ? rd("profiles") : nothing
    own === nothing || (own.subject = string.(own.subject))
    scale = medfin(unique(select(all.fi, :subject, :model, :scale)).scale)
    println(io, "## Package diagnostics\n")
    println(io, "The package's built-in tools, run on every model in the panel on the same ", d.n,
            " evenly spaced curves, each starting from its showcase fit: ",
            "`fisher_information`, `profile_likelihood`, `bootstrap_joint_fit` (residual, 40 resamples) and ",
            "`synthetic_recovery_benchmark` (20 simulated data sets per curve), plus 5-fold cross-validation. ",
            "Script: `showcase/analyses/package_checks.jl`.\n")
    println(io, "The 10% proportional error sets the weights; its overall size is estimated from each fit's residuals ",
            "(the same maximum-likelihood scale the BIC and the showcase profile use). Here the residual scatter is a median ",
            fmt(scale), "× the nominal 10%, so every interval and simulated data set is calibrated to the scatter the data really have.\n")
    if d.pairs > 0
        share = d.agree / d.pairs
        kind = share >= 0.75 ? "tip" : share >= 0.4 ? "warning" : "danger"
        label = share >= 0.75 ? "✅ Fisher and profile intervals agree for the pick" : share >= 0.4 ? "⚠️ Fisher and profile intervals partly agree for the pick" : "❌ Fisher intervals are not reliable for the pick"
        println(io, "!!! ", kind, " \"", label, "\"\n    For ", nice(pick), ", the Fisher (Wald) interval is within a factor of 1.5 of the profile interval for ", d.agree, " of ", d.pairs,
                " parameter-curve pairs. Where they disagree, trust the profile and bootstrap intervals: the Fisher interval assumes the likelihood is quadratic around the estimate.\n")
    end

    # every model, one row each
    mrows = Vector{String}[]
    for m in models
        x = checks_data(all, m)
        k = length(unique(x.fi.parameter))
        push!(mrows, [(m == pick ? "**" * nice(m) * "** (pick)" : nice(m)), string(k),
                      share_str(x.agree, x.pairs), pct(closed(x.pg)),
                      fmt(100 * medfin(relwidth.(x.bs.lower, x.bs.upper, x.bs.estimate))),
                      fmt(100 * medfin(abs.(x.rc.rel_bias))),
                      fmt(100 * medfin(x.cv.median_rel_error)), fmt(100 * medfin(x.cv.within_20pct)),
                      sum(x.cv.k) > 0 ? fmt(100 * sum(x.cv.folds_converged) / sum(x.cv.k)) : "—"])
    end
    println(io, "**Every model** (medians over curves and parameters)\n")
    println(io, md_table(["Model", "Parameters", "Fisher agrees with profile (%)", "Profile interval closes (%)",
                          "Bootstrap 95% width (% of estimate)", "Synthetic recovery: abs. bias (%)",
                          "Cross-validation: held-out error (%)", "Held-out points within 20% (%)", "Folds converged (%)"], mrows))
    println(io, "\n*Profile interval closes*: the 95% interval ends inside the parameter bounds on both sides. ",
            "*Cross-validation*: the first point is the initial condition and stays in every training set; the others are dealt into 5 interleaved folds, ",
            "each refitted with the showcase objective and predicted from the full curve, so this tests interpolation. A fold counts as converged when its fit reaches a weighted error per point within 4× the full-curve fit's (retried with up to 32 starts); folds that never do are not scored. The forecast check above tests extrapolation.\n")

    # the pick, parameter by parameter
    rows = Vector{String}[]
    for nm in unique(d.fi.parameter)
        f = d.fi[d.fi.parameter .== nm, :]
        g = d.pg[d.pg.parameter .== nm, :]
        b = d.bs[d.bs.parameter .== nm, :]
        r = d.rc[d.rc.parameter .== nm, :]
        wf = medfin(relwidth.(f.wald_lower, f.wald_upper, f.estimate))
        wp = medfin(relwidth.(g.lower, g.upper, g.estimate)[closed(g)])
        wb = medfin(relwidth.(b.lower, b.upper, b.estimate))
        wo = "—"
        if own !== nothing
            o = own[(own.parameter .== nm) .& in.(own.subject, Ref(Set(f.subject))), :]
            nrow(o) > 0 && (wo = fmt(medfin(relwidth.(o.lower, o.upper, o.estimate)[o.identifiable])) * " (" * pct(o.identifiable) * "%)")
        end
        push!(rows, [nm, fmt(wf), fmt(wp) * " (" * pct(closed(g)) * "%)", wo, fmt(wb),
                     fmt(100 * medfin(r.rel_bias)), fmt(100 * medfin(r.rel_rmse))])
    end
    println(io, "**The pick, ", nice(pick), ", parameter by parameter.** 95% interval width as median % of the estimate (in brackets: share of curves where the interval closes), and synthetic recovery.\n")
    println(io, md_table(["Parameter", "Fisher (Wald)", "Package profile", "Showcase profile", "Bootstrap",
                          "Recovery: median bias (% of truth)", "Recovery: median RMSE (% of truth)"], rows))
    println(io, "\nThe package profile uses the package's default grid plus points at ±0.5 to ±4 Fisher standard errors around the estimate. ",
            "`profile_likelihood` reports the outermost grid points inside the cutoff, which reads about 20% narrow at this spacing, so the widths here interpolate the crossing between grid points. ",
            "The showcase profile searches outward and bisects the crossing, and uses the exact log-likelihood ratio n·log(wSSE/wSSE₀) rather than its quadratic form, so it reads slightly wider.\n")
end

function cross_page(done)
    dir = joinpath(SHOWCASE, "results", "cross")
    isfile(joinpath(dir, "summary.csv")) || return nothing
    summ = CSV.read(joinpath(dir, "summary.csv"), DataFrame)
    summ = summ[in.(summ.dataset, Ref(Set(done))), :]
    io = IOBuffer()
    println(io, "# Across datasets\n")
    println(io, "The same panel, recipe and rule on every dataset, side by side. ",
            "Rebuilt from the committed results with `julia --project=showcase showcase/analyses/cross_dataset.jl`.\n")
    rows = [["[" * replace(r.dataset, "_" => " ") * "](" * r.dataset * ".md)", r.scale, string(r.n_curves),
             nice(r.pick), fmt(r.gap_bic), isnan(r.ident_share) ? "—" : string(round(Int, 100 * r.ident_share)),
             r.forecast_best isa AbstractString && !isempty(r.forecast_best) ? nice(r.forecast_best) * " (" * fmt(100 * r.forecast_best_err) * "%)" : "—",
             verdict_label(r.dataset)]
            for r in eachrow(summ)]
    println(io, md_table(["Dataset", "Scale", "Curves", "Rule pick", "ΔBIC to next model",
                          "Fully identifiable (%)", "Best forecaster (median error)", "Agrees with paper?"], rows))
    println(io, "\n*Fully identifiable*: share of curves on which every parameter of the pick has a closed 95% profile interval.\n")
    println(io, "\n## Which law wins where\n")
    println(io, "Each cell is the summed ΔBIC divided by the number of curves, so datasets of different size share one scale; the star marks the rule pick. Grey means the model was not in that dataset's panel.\n")
    println(io, "### Untreated growth\n")
    println(io, "![ΔBIC heatmap](../assets/cross/delta_bic_heatmap.png)\n")
    println(io, "How often each law is the pick when every curve is judged on its own. Short or noisy curves favour simple laws, which is why per-curve shares and the dataset-level pick can disagree.\n")
    println(io, "![Pick share heatmap](../assets/cross/pick_share_heatmap.png)\n")
    if isfile(joinpath(SHOWCASE, "figures", "cross", "delta_bic_heatmap_treated.png"))
        println(io, "### Under treatment\n")
        println(io, "Treated datasets use a panel of treatment-response laws (see each dataset page).\n")
        println(io, "![ΔBIC heatmap, treated](../assets/cross/delta_bic_heatmap_treated.png)\n")
        println(io, "![Pick share heatmap, treated](../assets/cross/pick_share_heatmap_treated.png)\n")
    end
    withchecks = [id for id in done if isfile(joinpath(SHOWCASE, "results", id, "checks_fisher.csv"))]
    if !isempty(withchecks)
        println(io, "## Package diagnostics by dataset\n")
        println(io, "The package's own tools on up to 12 curves per dataset, for every model in the panel (details on each page). ",
                "Fisher agrees = Wald interval within a factor of 1.5 of the profile interval; profile closes = the 95% interval ends inside the parameter bounds; ",
                "error scale = residual scatter as a multiple of the nominal 10% error.\n")
        crow = Vector{String}[]
        bymodel = DataFrame(model = String[], dataset = String[], agree = Int[], pairs = Int[], closes = Float64[], bias = Float64[], cv = Float64[])
        for id in withchecks
            res = joinpath(SHOWCASE, "results", id)
            all = read_checks(res)
            pick = TOML.parsefile(joinpath(res, "meta.toml"))["pick"]
            for m in unique(all.fi.model)
                x = checks_data(all, m)
                push!(bymodel, (m, id, x.agree, x.pairs, isempty(x.pg.status) ? NaN : count(closed(x.pg)) / nrow(x.pg),
                                medfin(abs.(x.rc.rel_bias)), medfin(x.cv.median_rel_error)))
            end
            d = checks_data(all, pick)
            sub = bymodel[bymodel.dataset .== id, :]
            bestcv = sub.model[argmin(replace(sub.cv, NaN => Inf))]
            push!(crow, ["[" * page_title(id) * "](" * id * ".md)", string(d.n), string(nrow(sub)),
                         fmt(medfin(unique(select(all.fi, :subject, :model, :scale)).scale)),
                         share_str(d.agree, d.pairs), pct(closed(d.pg)), fmt(100 * medfin(abs.(d.rc.rel_bias))),
                         fmt(100 * medfin(d.cv.median_rel_error)), nice(bestcv) * " (" * fmt(100 * minimum(replace(sub.cv, NaN => Inf))) * "%)"])
        end
        println(io, "**The pick on each dataset**\n")
        println(io, md_table(["Dataset", "Curves", "Models checked", "Error scale (× 10%)", "Fisher agrees (%)", "Profile closes (%)",
                              "Synthetic recovery: abs. bias (%)", "Cross-validation: held-out error (%)", "Lowest held-out error (model)"], crow))
        println(io, "\n**Each model, across the datasets it was fitted on** (medians of the per-dataset values)\n")
        mrow = Vector{String}[]
        for g in groupby(bymodel, :model)
            push!(mrow, [nice(g.model[1]), string(nrow(g)), share_str(sum(g.agree), sum(g.pairs)),
                         fmt(100 * medfin(g.closes)), fmt(100 * medfin(g.bias)), fmt(100 * medfin(g.cv))])
        end
        sort!(mrow, by = r -> r[1])
        println(io, md_table(["Model", "Datasets", "Fisher agrees (%)", "Profile closes (%)", "Synthetic recovery: abs. bias (%)",
                              "Cross-validation: held-out error (%)"], mrow))
        println(io)
    end
    if isfile(joinpath(SHOWCASE, "figures", "cross", "identifiability_by_dataset.png"))
        println(io, "## Identifiability by dataset\n")
        println(io, "Share of curves on which every parameter of the pick has a closed 95% profile interval.\n")
        println(io, "![Identifiability](../assets/cross/identifiability_by_dataset.png)\n")
    end
    println(io, "## Gallery\n")
    for id in done
        isfile(joinpath(SHOWCASE, "figures", id, "fit_gallery.png")) || continue
        println(io, "### [", page_title(id), "](", id, ".md)\n")
        println(io, "![", id, "](../assets/", id, "/fit_gallery.png)\n")
    end
    write(joinpath(GEN, "cross.md"), String(take!(io)))
    copy_figs("cross")
    return "datasets/cross.md"
end

function copy_figs(id)
    figs = joinpath(SHOWCASE, "figures", id)
    isdir(figs) || return
    dest = joinpath(SRC, "assets", id)
    mkpath(dest)
    for f in readdir(figs)
        endswith(f, ".png") && cp(joinpath(figs, f), joinpath(dest, f); force = true)
    end
end

using Statistics
mkpath(GEN)
done = sort([d for d in readdir(joinpath(SHOWCASE, "results")) if isfile(joinpath(SHOWCASE, "results", d, "meta.toml"))])
page_title(id) = get(TOML.parsefile(joinpath(SHOWCASE, "results", id, "meta.toml")), "title",
                     MANIFEST[get(TOML.parsefile(joinpath(SHOWCASE, "results", id, "meta.toml")), "source", id)]["title"])
pages = [page_title(id) => dataset_page(id) for id in done]
cross = cross_page(done)

"""One page listing every dataset's verdict against its paper."""
function agreement_page(done)
    io = IOBuffer()
    println(io, "# Does it match the papers?\n")
    println(io, "Each dataset page opens with a coloured box comparing the showcase result with the source paper. This page collects them. ",
            "The showcase applies one shared recipe to every dataset, so it re-analyses the data rather than reproducing each paper's own analysis; ",
            "where the two differ, the box says why.\n")
    counts = Dict(k => count(id -> get(get(VERDICTS, id, Dict()), "verdict", "") == k, done) for k in keys(VERDICT_STYLE))
    println(io, md_table(["Verdict", "Datasets"], [[VERDICT_STYLE[k][2], string(counts[k])]
                                                  for k in ("match", "partial", "differs", "not_comparable")]), "\n")
    rows = [["[" * page_title(id) * "](" * id * ".md)", verdict_label(id),
             get(get(VERDICTS, id, Dict()), "paper", "—"), get(get(VERDICTS, id, Dict()), "ours", "—")] for id in done]
    println(io, md_table(["Dataset", "Verdict", "Paper", "This showcase"], rows))
    write(joinpath(GEN, "agreement.md"), String(take!(io)))
    return "datasets/agreement.md"
end
agreement = agreement_page(done)

makedocs(;
    sitename = "GrowthParameterEstimation.jl showcase",
    authors = "Kadin El bakkouri",
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true", edit_link = "main", inventory_version = "showcase",
                             canonical = "https://kadinelbak.github.io/GrowthParameterEstimation.jl/showcase/"),
    pages = ["Home" => "index.md", "Does it match the papers?" => agreement, "What it covers" => "coverage.md",
             "Why one package" => "why.md", "Shared recipe" => "recipe.md",
             (cross === nothing ? [] : ["Across datasets" => cross])..., "Datasets" => pages, "Reproduce" => "reproduce.md"],
    warnonly = true,
)

# Deploy beside the package docs at https://kadinelbak.github.io/GrowthParameterEstimation.jl/showcase/
# (only from CI on main; a local build just writes showcase/docs/build/).
deploydocs(;
    repo = "github.com/kadinelbak/GrowthParameterEstimation.jl",
    devbranch = "main",
    dirname = "showcase",
    versions = nothing,
    push_preview = false,
)
