# Build the showcase site from committed results (no fitting happens here).
#   julia --project=showcase showcase/docs/make.jl
using Documenter, CSV, DataFrames, TOML, Printf

const SHOWCASE = normpath(joinpath(@__DIR__, ".."))
const SRC = joinpath(@__DIR__, "src")
const GEN = joinpath(SRC, "datasets")
const MANIFEST = TOML.parsefile(joinpath(SHOWCASE, "data", "manifest.toml"))["datasets"]

nice(m) = replace(replace(m, "_growth" => ""), "_" => " ")
fmt(x::Real) = isfinite(x) ? @sprintf("%.1f", x) : "—"
fmtp(x::Real) = !isfinite(x) ? "—" : (abs(x) >= 1e4 || abs(x) < 1e-2) ? @sprintf("%.3g", x) : @sprintf("%.3f", x)

function md_table(header, rows)
    io = IOBuffer()
    println(io, "| ", join(header, " | "), " |")
    println(io, "|", join(fill(" --- ", length(header)), "|"), "|")
    for r in rows
        println(io, "| ", join(r, " | "), " |")
    end
    return String(take!(io))
end

function dataset_page(id)
    res = joinpath(SHOWCASE, "results", id)
    meta = TOML.parsefile(joinpath(res, "meta.toml"))
    src = MANIFEST[id]
    rd(n) = CSV.read(joinpath(res, "$(n).csv"), DataFrame)
    ranking = rd("ranking")
    io = IOBuffer()
    println(io, "# ", src["title"], "\n")
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
    println(io, "## Reproduce\n")
    println(io, "```bash\npython3 -I showcase/data/prepare.py --only ", id, "\njulia --project=showcase -t auto showcase/run_pipeline.jl --only ", id, "\n```\n")
    println(io, "Fitted with GrowthParameterEstimation.jl ", meta["package_version"], ". Results: `showcase/results/", id, "/`.")
    path = joinpath(GEN, "$(id).md")
    write(path, String(take!(io)))
    # copy figures next to the page
    figs = joinpath(SHOWCASE, "figures", id)
    if isdir(figs)
        dest = joinpath(SRC, "assets", id)
        mkpath(dest)
        for f in readdir(figs)
            endswith(f, ".png") && cp(joinpath(figs, f), joinpath(dest, f); force = true)
        end
    end
    return "datasets/$(id).md"
end

using Statistics
mkpath(GEN)
done = sort([d for d in readdir(joinpath(SHOWCASE, "results")) if isfile(joinpath(SHOWCASE, "results", d, "meta.toml"))])
pages = [MANIFEST[id]["title"] => dataset_page(id) for id in done]

makedocs(;
    sitename = "GrowthParameterEstimation.jl showcase",
    authors = "Kadin El bakkouri",
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true",
                             canonical = "https://kadinelbak.github.io/GrowthParameterEstimation.jl/showcase/"),
    pages = ["Home" => "index.md", "Shared recipe" => "recipe.md", "Datasets" => pages, "Reproduce" => "reproduce.md"],
    warnonly = true,
)
