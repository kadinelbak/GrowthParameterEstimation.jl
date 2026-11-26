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

"""Read the package-diagnostics tables (written by analyses/package_checks.jl) for one dataset,
with the Fisher-vs-profile agreement count."""
function checks_data(res)
    rd(n) = (d = CSV.read(joinpath(res, "$(n).csv"), DataFrame); d.subject = string.(d.subject); d)
    fi, pr, bs, rc, cv = rd("checks_fisher"), rd("checks_profile"), rd("checks_boot"), rd("checks_recovery"), rd("checks_cv")
    pg = pr[pr.grid .== "fine_grid", :]
    pd = pr[pr.grid .== "default_grid", :]
    pg2 = copy(pg)
    pg2.closed = closed(pg)
    j = innerjoin(select(fi, :subject, :parameter, [:wald_lower, :wald_upper, :estimate] => ByRow(relwidth) => :wf),
                  select(pg2, :subject, :parameter, :closed, [:lower, :upper, :estimate] => ByRow(relwidth) => :wp),
                  on = [:subject, :parameter])
    ok = j[j.closed .& isfinite.(j.wf) .& isfinite.(j.wp), :]
    agree = isempty(ok) ? 0 : count(r -> 2 / 3 <= r.wf / r.wp <= 1.5, eachrow(ok))
    return (; fi, pg, pd, bs, rc, cv, n = length(unique(fi.subject)), agree, pairs = nrow(j))
end

"""The package's own diagnostics on a subset of curves, as a dataset-page section."""
function package_checks_section(io, id, res, rd)
    (; fi, pg, pd, bs, rc, cv, n, agree, pairs) = checks_data(res)
    own = isfile(joinpath(res, "profiles.csv")) ? rd("profiles") : nothing
    own === nothing || (own.subject = string.(own.subject))
    println(io, "## Package diagnostics\n")
    println(io, "The package's built-in tools, run on ", n, " evenly spaced curves with the pick (",
            nice(TOML.parsefile(joinpath(res, "meta.toml"))["pick"]), ") and the same 10% proportional error: ",
            "`fisher_information`, `profile_likelihood`, `bootstrap_joint_fit` (residual, 40 resamples), ",
            "`synthetic_recovery_benchmark` (20 simulated data sets per curve) and `k_fold_cross_validation` (5 folds). ",
            "Script: `showcase/analyses/package_checks.jl`.\n")
    if pairs > 0
        share = agree / pairs
        kind = share >= 0.75 ? "tip" : share >= 0.4 ? "warning" : "danger"
        label = share >= 0.75 ? "✅ Fisher and profile intervals agree" : share >= 0.4 ? "⚠️ Fisher and profile intervals partly agree" : "❌ Fisher intervals are not reliable here"
        println(io, "!!! ", kind, " \"", label, "\"\n    The Fisher (Wald) interval is within a factor of 1.5 of the profile interval for ", agree, " of ", pairs,
                " parameter-curve pairs. Where they disagree, trust the profile and bootstrap intervals: the Fisher interval assumes the likelihood is quadratic around the estimate.\n")
    end

    rows = Vector{String}[]
    for nm in unique(fi.parameter)
        f = fi[fi.parameter .== nm, :]
        g = pg[pg.parameter .== nm, :]
        d = pd[pd.parameter .== nm, :]
        b = bs[bs.parameter .== nm, :]
        wf = medfin(relwidth.(f.wald_lower, f.wald_upper, f.estimate))
        wp = medfin(relwidth.(g.lower, g.upper, g.estimate)[closed(g)])
        wb = medfin(relwidth.(b.lower, b.upper, b.estimate))
        wo = "—"
        if own !== nothing
            o = own[(own.parameter .== nm) .& in.(own.subject, Ref(Set(f.subject))), :]
            nrow(o) > 0 && (wo = fmt(medfin(relwidth.(o.lower, o.upper, o.estimate)[o.identifiable])) * " (" * pct(o.identifiable) * "%)")
        end
        push!(rows, [nm, fmt(wf), fmt(wp) * " (" * pct(closed(g)) * "%)", wo, fmt(wb), pct(closed(d))])
    end
    println(io, "**95% interval width, median % of the estimate** (in brackets: share of curves where the interval closes inside the parameter bounds)\n")
    println(io, md_table(["Parameter", "Fisher (Wald)", "Package profile", "Showcase profile", "Bootstrap", "Package profile, default grid: closes (%)"], rows))
    println(io, "\nThe package profile here uses the package's default grid plus points at ±0.5 to ±4 Fisher standard errors around the estimate. ",
            "It reports the outermost grid points inside the cutoff, so it can read up to one grid step narrower than the showcase profile, which bisects the crossing. ",
            "Its default grid (8 log-spaced points per side spanning the whole parameter bounds) is too coarse for these curves: ",
            "often only the estimate passes the cutoff (a zero-width interval), or the interval runs to a bound. The package labels both `bounded`; the last column counts only intervals that really close.\n")

    rrows = Vector{String}[]
    for nm in unique(rc.parameter)
        r = rc[rc.parameter .== nm, :]
        push!(rrows, [nm, fmt(100 * medfin(r.rel_bias)), fmt(100 * medfin(r.rel_rmse)), fmt(100 * medfin(r.success_rate))])
    end
    println(io, "**Synthetic recovery** (data simulated from each curve's fit at its own time points and 10% noise, then refitted)\n")
    println(io, md_table(["Parameter", "Median bias (% of truth)", "Median RMSE (% of truth)", "Fits converged (%)"], rrows))
    if any(isfinite, cv.r_squared)
        println(io, "\n**Cross-validation.** Five-fold cross-validation on ", count(isfinite, cv.r_squared), " curves, with points held out at random (interpolation, unweighted): median R² ",
                @sprintf("%.2f", medfin(cv.r_squared)), ", median RMSE ", fmt(100 * medfin(cv.rel_rmse)),
                "% of the mean value. The forecast check above is the harder test, because it holds out the end of each curve.\n")
    else
        println(io, "\n**Cross-validation.** `k_fold_cross_validation` did not finish on this dataset: with a dose schedule its default solver fails, and with `Tsit5` the bounded Nelder-Mead fits ran for over an hour on 12 curves.\n")
    end
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
        println(io, "The package's own tools on up to 12 curves per dataset (details on each page). ",
                "Fisher agrees = Wald interval within a factor of 1.5 of the profile interval; profile closes = the 95% interval ends inside the parameter bounds.\n")
        crow = Vector{String}[]
        for id in withchecks
            d = checks_data(joinpath(SHOWCASE, "results", id))
            push!(crow, ["[" * page_title(id) * "](" * id * ".md)", string(d.n),
                         d.pairs == 0 ? "—" : string(round(Int, 100 * d.agree / d.pairs)),
                         pct(closed(d.pg)), pct(closed(d.pd)),
                         fmt(100 * medfin(abs.(d.rc.rel_bias))), (x = medfin(d.cv.r_squared); isfinite(x) ? @sprintf("%.2f", x) : "—")])
        end
        println(io, md_table(["Dataset", "Curves", "Fisher agrees (%)", "Profile closes (%)", "Default-grid profile closes (%)",
                              "Synthetic recovery: median abs. bias (%)", "k-fold CV median R²"], crow))
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
