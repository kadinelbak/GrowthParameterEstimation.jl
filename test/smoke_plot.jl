using GrowthParameterEstimation
using DataFrames

df = DataFrame(
    time        = vcat(collect(0.0:1.0:6.0), collect(0.0:1.0:6.0)),
    count       = vcat([1.0,1.5,2.2,3.0,3.8,4.5,5.1], [1.0,1.3,1.9,2.5,3.1,3.6,4.0]),
    error       = fill(0.2, 14),
    dose        = vcat(fill(0.2, 7), fill(0.6, 7)),
    cell_line   = fill("A549", 14),
    density     = fill(1.0, 14),
    replicate   = vcat(fill(1, 7), fill(2, 7)),
    unit_time   = fill("h", 14),
    unit_count  = fill("count", 14),
)

conditions = build_conditions(df)
@assert !isempty(conditions) "conditions must not be empty"

ranked = rank_models(
    ["logistic_growth", "theta_logistic_hill_inhibition"],
    conditions;
    n_starts=2, maxiters=60, top_k=2, seed=7,
)

# ── same filter used in the hardened GUI _rank_and_plot ──────────────────────
successful_models = [
    m for m in ranked.ranking.model
    if haskey(ranked.fits, m) && any(
        pc -> pc.success && !isempty(pc.observed) && all(isfinite, pc.observed),
        ranked.fits[m].per_condition
    )
]

if isempty(successful_models)
    println("SMOKE_FAIL: no successful models with finite predictions")
    exit(1)
end

selected_model = successful_models[1]
fit_info       = ranked.fits[selected_model]
selected_cond  = conditions[1].name

hit = findfirst(
    pc -> lowercase(strip(pc.condition)) == lowercase(strip(selected_cond)) && pc.success,
    fit_info.per_condition
)

if isnothing(hit)
    println("SMOKE_FAIL: no per-condition hit for $(selected_cond) in $(selected_model)")
    exit(1)
end

pred     = Float64.(fit_info.per_condition[hit].observed)
observed = conditions[1].count
t        = conditions[1].time

valid = length(pred) == length(observed) == length(t) &&
        !isempty(pred) &&
        all(isfinite, pred) && all(isfinite, observed) && all(isfinite, t)

if !valid
    println("SMOKE_FAIL: plot vectors invalid — lengths pred=$(length(pred)) obs=$(length(observed)) t=$(length(t)) finite=$(all(isfinite,pred))")
    exit(1)
end

println("SMOKE_OK: model=$(selected_model) cond=$(selected_cond) n=$(length(t)) points ready to plot")
