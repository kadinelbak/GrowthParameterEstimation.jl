# Shared recipe for every showcase dataset.
#
# One set of rules, applied the same way to cells, spheroids, mice and patients:
#   1. each curve is fitted separately with every model in the panel (`fit`, multistart);
#   2. models are scored by AIC, AICc and BIC; the pick is the model with the fewest
#      parameters within 2 BIC of the best (ties go to the lower BIC);
#   3. the pick gets profile-likelihood 95% intervals (refits with one parameter fixed);
#   4. a forecast check refits on the early points and predicts the rest.
#
# Error model: proportional, sigma = CV * max(y, floor), floor = 5% of the curve's maximum.
# The first observation is the initial condition, so no initial value is estimated.

using GrowthParameterEstimation
using CSV, DataFrames, Statistics, TOML, Printf

const SHOWCASE = normpath(joinpath(@__DIR__, ".."))
const TIDY = joinpath(SHOWCASE, "data", "tidy")
const RESULTS = joinpath(SHOWCASE, "results")

const GROWTH_PANEL = [
    "exponential_growth",
    "logistic_growth",
    "classic_gompertz_growth",
    "theta_logistic_growth",
    "bertalanffy_growth",
    "general_bertalanffy_growth",
    "power_law_growth",
    "exponential_linear_growth",
]

const RULE_BIC = 2.0
const CHI2_95 = 3.841458820694124
const CV = 0.1
# Bounded BFGS through an unconstrained transform: same optima as Fminbox here, far faster.
const OPTIMIZER = :transformed

"""Load one tidy dataset written by data/prepare.py."""
load_tidy(id::AbstractString) = CSV.read(joinpath(TIDY, "$(id).csv"), DataFrame)

"""Turn each subject in a tidy table into a `FitCondition` (one curve each)."""
function curves(df::DataFrame; min_points::Int = 5, value_floor_frac::Float64 = 0.05,
                exposure_for = _ -> ConstantExposure(0.0))
    out = FitCondition[]
    for g in groupby(sort(df, [:subject, :time]), :subject)
        g = g[.!ismissing.(g.value) .& .!ismissing.(g.time), :]
        nrow(g) < min_points && continue
        # average replicate measurements taken at the same time
        agg = combine(groupby(g, :time), :value => mean => :value)
        nrow(agg) < min_points && continue
        t = Float64.(agg.time)
        y = Float64.(agg.value)
        floor_ = value_floor_frac * maximum(y)
        sigma = CV .* max.(y, floor_)
        y[1] > 0 || continue
        meta = Dict{Symbol,Any}(:subject => string(g.subject[1]), :condition => string(g.condition[1]))
        push!(out, FitCondition(string(g.subject[1]), t, y, sigma, [y[1]], exposure_for(g), meta))
    end
    return out
end

aicc(aic, k, n) = n - k - 1 > 0 ? aic + 2k * (k + 1) / (n - k - 1) : Inf

"""Fit every model in `models` to every curve; return one row per (curve, model).
Curves are spread over threads (`julia -t auto`)."""
function fit_panel(conds::Vector{FitCondition}, models::Vector{String}; n_starts::Int = 8, maxiters::Int = 600)
    per_curve = Vector{Vector{NamedTuple}}(undef, length(conds))
    Threads.@threads for i in eachindex(conds)
        c = conds[i]
        rows = NamedTuple[]
        for m in models
            spec = get_model(m)
            n = length(c.time)
            try
                r = fit(spec, [c]; n_starts = n_starts, maxiters = maxiters, top_k = 1, optimizer = OPTIMIZER)
                k = r.n_params
                push!(rows, (subject = c.name, condition = string(c.metadata[:condition]), model = m, n = n, k = k,
                             wsse = r.best.objective, aic = r.best.aic, aicc = aicc(r.best.aic, k, n), bic = r.best.bic,
                             params = join(string.(round.(r.best.params; sigdigits = 6)), ";")))
            catch err
                push!(rows, (subject = c.name, condition = string(c.metadata[:condition]), model = m, n = n,
                             k = length(spec.param_names), wsse = Inf, aic = Inf, aicc = Inf, bic = Inf, params = ""))
            end
        end
        per_curve[i] = rows
    end
    return DataFrame(reduce(vcat, per_curve))
end

"""Apply the selection rule within each group of rows (a curve, or a whole dataset)."""
function rule_pick(sub::AbstractDataFrame)
    best = minimum(sub.bic)
    near = sub[sub.bic .<= best + RULE_BIC, :]
    kmin = minimum(near.k)
    cand = near[near.k .== kmin, :]
    return cand.model[argmin(cand.bic)]
end

"""Per-curve picks and a dataset-level ranking (BIC summed over curves)."""
function summarise_panel(panel::DataFrame)
    ok = panel[isfinite.(panel.bic), :]
    per_curve = combine(groupby(ok, :subject), AsTable(:) => (t -> rule_pick(DataFrame(t))) => :pick,
                        :bic => minimum => :best_bic)
    lowest = combine(groupby(ok, :subject), [:model, :bic] => ((m, b) -> m[argmin(b)]) => :lowest_bic_model)
    per_curve = innerjoin(per_curve, lowest, on = :subject)
    # Only curves where every model fitted enter the dataset totals.
    complete = combine(groupby(ok, :subject), nrow => :nm)
    keep = complete.subject[complete.nm .== length(unique(panel.model))]
    tot = combine(groupby(ok[in.(ok.subject, Ref(Set(keep))), :], :model),
                  :k => first => :k, :aic => sum => :aic, :aicc => sum => :aicc, :bic => sum => :bic)
    sort!(tot, :bic)
    tot.delta_bic = tot.bic .- minimum(tot.bic)
    tot.delta_aic = tot.aic .- minimum(tot.aic)
    wins = combine(groupby(per_curve, :pick), nrow => :curves_picked)
    tot = leftjoin(tot, rename(wins, :pick => :model), on = :model)
    tot.curves_picked = coalesce.(tot.curves_picked, 0)
    sort!(tot, :bic)
    pick = rule_pick(rename(select(tot, :model, :k, :bic), :bic => :bic))
    return (per_curve = per_curve, ranking = tot, pick = pick, n_curves = length(keep))
end

"""Profile-likelihood 95% interval for each parameter of `model` on one curve.

Each parameter is held fixed with `fit(...; fixed_params)` while the others are refitted.
From the estimate the search steps outward (on a log scale for positive parameters, with
growing steps) until n*log(WSSE/n) rises by the chi-square(1) 95% cutoff or the bound is
reached, then bisects the crossing. An interval that reaches a bound is open.
"""
function profile_intervals(c::FitCondition, model::String; n_starts::Int = 3, bisections::Int = 6)
    spec = get_model(model)
    best = fit(spec, [c]; n_starts = 8, top_k = 1, optimizer = OPTIMIZER)
    phat = best.best.params
    n = length(c.time)
    base = n * log(max(best.best.objective, 1e-12) / n)
    single = length(spec.param_names) == 1
    function prof(name, v)
        if single
            sim = simulate(spec, c.time, [v]; u0 = c.u0)
            w = sim.success ? sum(((c.count .- sim.observed) ./ c.error) .^ 2) : Inf
            return n * log(max(w, 1e-12) / n) - base
        end
        r = try
            fit(spec, [c]; fixed_params = Dict(name => v), n_starts = n_starts, top_k = 1, optimizer = OPTIMIZER)
        catch
            nothing
        end
        return r === nothing ? Inf : n * log(max(r.best.objective, 1e-12) / n) - base
    end
    rows = NamedTuple[]
    for (j, name) in enumerate(spec.param_names)
        lo, hi = spec.bounds[j]
        logscale = lo > 0 && phat[j] > 0
        to(x) = logscale ? log(x) : x
        from(z) = logscale ? exp(z) : z
        zhat, zlo, zhi = to(phat[j]), to(lo), to(hi)
        span = zhi - zlo
        ends = map((-1, 1)) do dir
            limit = dir < 0 ? zlo : zhi
            step = span / 200
            inside = zhat
            crossed = false
            while true
                z = clamp(inside + dir * step, min(zlo, zhi), max(zlo, zhi))
                if prof(name, from(z)) > CHI2_95
                    outside = z
                    for _ in 1:bisections
                        mid = (inside + outside) / 2
                        prof(name, from(mid)) > CHI2_95 ? (outside = mid) : (inside = mid)
                    end
                    crossed = true
                    break
                end
                inside = z
                z == limit && break
                step *= 2
            end
            (from(inside), crossed)
        end
        (lower, lo_closed), (upper, hi_closed) = ends
        push!(rows, (parameter = string(name), estimate = phat[j], lower = lower, upper = upper,
                     identifiable = lo_closed && hi_closed))
    end
    return DataFrame(rows)
end

"""Fit on the first `frac` of each curve and report the error on the remaining points."""
function forecast_check(conds::Vector{FitCondition}, models::Vector{String}; frac::Float64 = 0.7, n_starts::Int = 6)
    out = Vector{Vector{NamedTuple}}(undef, length(conds))
    Threads.@threads for i in eachindex(conds)
        c = conds[i]
        rows = NamedTuple[]
        out[i] = rows
        n = length(c.time)
        ntrain = max(3, floor(Int, frac * n))
        ntrain >= n && continue
        train = FitCondition(c.name, c.time[1:ntrain], c.count[1:ntrain], c.error[1:ntrain], c.u0, c.exposure, c.metadata)
        for m in models
            spec = get_model(m)
            r = try
                fit(spec, [train]; n_starts = n_starts, top_k = 1, optimizer = OPTIMIZER)
            catch
                nothing
            end
            r === nothing && continue
            sim = simulate(spec, c.time, r.best.params; u0 = c.u0)
            sim.success || continue
            relerr = abs.(sim.observed[ntrain+1:end] .- c.count[ntrain+1:end]) ./ max.(c.count[ntrain+1:end], 1e-12)
            push!(rows, (subject = c.name, model = m, n_train = ntrain, n_test = n - ntrain,
                         median_rel_error = median(relerr), within_20pct = mean(relerr .<= 0.2)))
        end
    end
    return DataFrame(reduce(vcat, out))
end

"""Profile intervals of `model` on every curve (threaded)."""
function profile_all(conds::Vector{FitCondition}, model::String)
    out = Vector{DataFrame}(undef, length(conds))
    Threads.@threads for i in eachindex(conds)
        out[i] = insertcols!(profile_intervals(conds[i], model), 1, :subject => conds[i].name)
    end
    return reduce(vcat, out)
end

"""Write a dataset's results under results/<id>/."""
function save_results(id::AbstractString; kwargs...)
    dir = joinpath(RESULTS, id)
    mkpath(dir)
    for (name, tbl) in kwargs
        if tbl isa DataFrame
            CSV.write(joinpath(dir, "$(name).csv"), tbl)
        else
            open(joinpath(dir, "$(name).toml"), "w") do io
                TOML.print(io, tbl)
            end
        end
    end
    return dir
end
