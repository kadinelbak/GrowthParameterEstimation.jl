# Benzekry et al. 2014, Lewis lung carcinoma: 20 mice, untreated, volumes in mm3.
# Published comparison: Gompertz and power law were the most parsimonious, identifiable fits.
include(joinpath(@__DIR__, "common.jl"))

const ID = "benzekry2014_llc"
df = load_tidy(ID)
conds = curves(df; min_points = 5)
@info "$(ID): $(length(conds)) curves"

panel = fit_panel(conds, GROWTH_PANEL)
summary = summarise_panel(panel)
println(summary.ranking)
println("Rule pick: ", summary.pick)

# Profile intervals for the pick on every curve
profiles = vcat([insertcols!(profile_intervals(c, summary.pick), 1, :subject => c.name) for c in conds]...)

# Forecast: fit the first 70% of points, predict the rest
forecast = forecast_check(conds, GROWTH_PANEL)
fc = combine(groupby(forecast, :model), :median_rel_error => median => :median_rel_error,
             :within_20pct => mean => :share_within_20pct)
sort!(fc, :median_rel_error)
println(fc)

save_results(ID; panel = panel, per_curve = summary.per_curve, ranking = summary.ranking,
             profiles = profiles, forecast = forecast, forecast_summary = fc,
             meta = Dict("dataset" => ID, "pick" => summary.pick, "n_curves" => summary.n_curves,
                         "rule" => "fewest parameters within $(RULE_BIC) BIC of best",
                         "package_version" => string(pkgversion(GrowthParameterEstimation))))
