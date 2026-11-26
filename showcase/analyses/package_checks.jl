# The package's own diagnostics on every candidate model: Fisher information, the built-in
# profile likelihood, residual bootstrap, synthetic recovery, and k-fold cross-validation.
#
#   julia --project=showcase -t auto showcase/analyses/package_checks.jl --only <id> [--curves 12] [--cv-only]
#
# For each dataset, every model in the panel is checked on the same evenly spaced subset of
# curves, starting from the showcase fit. The weighted objective is reproduced exactly by
# fitting y/σ with an observable u -> u/σ(t). The 10% error is only a weighting: its overall
# scale is estimated from each fit's residuals, s² = wSSE/n (the maximum-likelihood value the
# showcase's BIC and profile already use), and passed as `residual_scale`, so Fisher, profile
# and recovery are calibrated to the scatter the data actually have.
# Writes results/<id>/checks_*.csv (one row per curve × model × parameter).

isdefined(@__MODULE__, :dataset_conditions) || include(joinpath(@__DIR__, "..", "run_pipeline.jl"))
using Random

const N_BOOT = 40
const N_RECOVERY = 20
const K_FOLDS = 5
const MAXITERS = 3000
const CV_RETRY = 4.0   # fold fit counts as converged when its wSSE/n is within 4x the full fit's

parse_params(s) = parse.(Float64, split(string(s), ";"))

"""Wrap a registry model as the `(du, u, p, t)` function the joint-fit tools expect."""
function plain_rhs(spec, exposure)
    takes_exposure = applicable(spec.ode!, zeros(1), ones(1), ones(length(spec.param_names)), 0.0, exposure)
    return takes_exposure ? ((du, u, p, t) -> spec.ode!(du, u, p, t, exposure)) : ((du, u, p, t) -> spec.ode!(du, u, p, t))
end

"""One curve as a weighted joint-fit dataset: y/σ against u/σ(t), with residual scale `s`
(the estimated multiplier on the nominal 10% error)."""
function weighted_spec(c::FitCondition, s::Float64 = 1.0)
    sigma = Dict(zip(c.time, c.error))
    return [(x = c.time, y = c.count ./ c.error, observable = (u, p, t) -> u[1] / sigma[t], residual_scale = s)]
end

"""Profile grid: ±0.5…4 Fisher standard errors around `est` (±5…300% when the error is
missing), plus the package's default log-spaced grid over the whole bounds, so the interval
can close whether or not the Fisher error is a good guide."""
function fine_grid(est, se, bnd)
    g = isfinite(se) && se > 0 ? est .+ se .* vcat(-4:0.5:-0.5, 0.5:0.5:4) :
        est .* exp.(vcat(-log(4):log(4)/8:-log(4)/8, log(4)/8:log(4)/8:log(4)))
    coarse = GrowthParameterEstimation.Identifiability._profile_grid(est, bnd[1], bnd[2], 8)
    return sort(unique(vcat(est, coarse, filter(v -> bnd[1] <= v <= bnd[2], g))))
end

"""k-fold cross-validation with the showcase's own objective: the first point stays in every
training set (it is the initial condition), the others are dealt into `k` interleaved folds,
each fold is refitted with the weighted `fit` and predicted from the full-curve simulation.

`s2` is the full-curve fit's wSSE/n. A fold fit whose training wSSE per point is more than
`CV_RETRY`× that has missed the optimum the full data found (random starts can stall in a
corner of the bounds, e.g. theta-logistic with r → 0, K → upper bound); it is refitted with
more starts and a new seed, up to three times. Folds that still miss are not scored, and
`folds_converged` says how many were."""
function showcase_cv(c::FitCondition, spec; k = K_FOLDS, s2 = Inf)
    n = length(c.time)
    k = min(k, n - 1 - length(spec.param_names))   # every training set keeps > k_params points
    k < 2 && return (median_rel_error = NaN, within_20pct = NaN, folds_ok = 0, folds_converged = 0, k = 0)
    floor_ = 0.05 * maximum(c.count)
    errs = Float64[]
    ok = 0
    converged = 0
    for f in 1:k
        test = [i for i in 2:n if (i - 2) % k + 1 == f]
        train = setdiff(1:n, test)
        tc = FitCondition(c.name, c.time[train], c.count[train], c.error[train], c.u0, c.exposure, c.metadata)
        r = nothing
        for (attempt, ns) in enumerate((8, 16, 32))
            cand = try
                fit(spec, [tc]; n_starts = ns, top_k = 1, optimizer = OPTIMIZER, seed = 42 + attempt)
            catch
                nothing
            end
            cand === nothing && continue
            (r === nothing || cand.best.objective < r.best.objective) && (r = cand)
            r.best.objective / length(train) <= CV_RETRY * s2 && break
        end
        r === nothing && continue
        r.best.objective / length(train) <= CV_RETRY * s2 || continue
        converged += 1
        sim = simulate(spec, c.time, r.best.params; u0 = c.u0, exposure = c.exposure)
        sim.success || continue
        ok += 1
        append!(errs, abs.(sim.observed[test] .- c.count[test]) ./ max.(c.count[test], floor_))
    end
    return isempty(errs) ? (median_rel_error = NaN, within_20pct = NaN, folds_ok = ok, folds_converged = converged, k = k) :
        (median_rel_error = median(errs), within_20pct = mean(errs .<= 0.2), folds_ok = ok, folds_converged = converged, k = k)
end

"""Where the profile crosses the cutoff on each side of `est`, interpolated linearly between
the last grid point inside and the first one outside (NaN when no point outside exists).
`profile_likelihood` itself reports the last point inside, which reads about one grid step
narrow."""
function crossing(rows, est, threshold)
    rows = sort(rows[isfinite.(rows.delta_sse), :], :fixed_value)
    x, y = rows.fixed_value, max.(rows.delta_sse, 0.0)
    i0 = argmin(abs.(x .- est))
    side(range) = begin
        prev = i0
        for i in range
            if y[i] > threshold
                return x[prev] + (threshold - y[prev]) / (y[i] - y[prev]) * (x[i] - x[prev])
            end
            prev = i
        end
        NaN
    end
    return side(i0 - 1:-1:1), side(i0 + 1:length(x))
end

function check_curve(c::FitCondition, model::String, p::Vector{Float64}, s::Float64; seed = 1)
    spec = get_model(model)
    rhs = plain_rhs(spec, c.exposure)
    ds = weighted_spec(c, s)
    names = spec.param_names
    bounds = [(Float64(lo), Float64(hi)) for (lo, hi) in spec.bounds]
    rng = MersenneTwister(seed)
    tag = (subject = c.name, model = model)
    out = (fisher = NamedTuple[], profile = NamedTuple[], boot = NamedTuple[], recovery = NamedTuple[], cv = NamedTuple[])

    # 1. Fisher information at the showcase estimate (local, Wald-type intervals)
    fi = try
        fisher_information(rhs, ds, c.u0, p; parameter_names = names)
    catch
        nothing
    end
    for (j, nm) in enumerate(names)
        se = fi === nothing ? NaN : fi.table.local_std_error[j]
        push!(out.fisher, (; tag..., parameter = String(nm), estimate = p[j], scale = s, std_error = se,
                           wald_lower = p[j] - 1.96se, wald_upper = p[j] + 1.96se,
                           full_rank = fi !== nothing && fi.numerical_rank == length(p),
                           condition_number = fi === nothing ? NaN : fi.condition_number))
    end

    # 2. Built-in profile likelihood on the package's default grid plus ±0.5…4 Fisher standard
    #    errors around the estimate (`fine_grid`); the default grid alone is too coarse.
    grid(j) = fine_grid(p[j], fi === nothing ? NaN : fi.table.local_std_error[j], bounds[j])
    pr = try
        profile_likelihood(rhs, ds, c.u0, p; bounds, parameter_names = names, points_per_side = 8,
                           values = Dict(names[j] => grid(j) for j in eachindex(names)),
                           optimizer = :nelder_mead, maxiters = MAXITERS)
    catch
        nothing
    end
    for (j, nm) in enumerate(names)
        r = pr === nothing ? nothing : pr.confidence_intervals[j, :]
        ilo, ihi = pr === nothing ? (NaN, NaN) : crossing(pr.profile[pr.profile.parameter .== String(nm), :], p[j], pr.threshold)
        push!(out.profile, (; tag..., grid = "fine_grid", parameter = String(nm), estimate = p[j],
                            lower = r === nothing ? NaN : r.lower, upper = r === nothing ? NaN : r.upper,
                            lower_interp = ilo, upper_interp = ihi,
                            status = r === nothing ? "failed" : r.confidence_status,
                            grid_min = bounds[j][1], grid_max = bounds[j][2],
                            # the baseline refit should not move: a large change means the
                            # profile is centred on a different optimum than the showcase fit
                            baseline_shift = pr === nothing ? NaN : abs(pr.fit.params[j] - p[j]) / max(abs(p[j]), 1e-12)))
    end

    # 3. Residual bootstrap
    bs = try
        bootstrap_joint_fit(rhs, ds, c.u0, p; bounds, n_bootstrap = N_BOOT, method = :residual, rng,
                            optimizer = :nelder_mead, maxiters = MAXITERS)
    catch
        nothing
    end
    for (j, nm) in enumerate(names)
        push!(out.boot, (; tag..., parameter = String(nm), estimate = p[j],
                         lower = bs === nothing ? NaN : bs.summary.ci_lower[j],
                         upper = bs === nothing ? NaN : bs.summary.ci_upper[j],
                         success_rate = bs === nothing ? 0.0 : bs.success_rate))
    end

    # 4. Synthetic recovery: simulate from the estimate with the estimated error, refit
    rc = try
        synthetic_recovery_benchmark(rhs, ds, c.u0, p; bounds, n_simulations = N_RECOVERY, rng,
                                     optimizer = :nelder_mead, maxiters = MAXITERS)
    catch
        nothing
    end
    for (j, nm) in enumerate(names)
        push!(out.recovery, (; tag..., parameter = String(nm), truth = p[j],
                             rel_bias = rc === nothing ? NaN : rc.summary.bias[j] / abs(p[j]),
                             rel_rmse = rc === nothing ? NaN : rc.summary.rmse[j] / abs(p[j]),
                             success_rate = rc === nothing ? 0.0 : rc.success_rate))
    end

    # 5. k-fold cross-validation on the showcase objective
    cv = showcase_cv(c, spec; s2 = s^2)
    push!(out.cv, (; tag..., cv...))
    return out
end

function package_checks(id::AbstractString, conds::Vector{FitCondition}; max_curves = 12, cv_only = false)
    dir = joinpath(RESULTS, id)
    panel = CSV.read(joinpath(dir, "panel.csv"), DataFrame)
    panel.subject = string.(panel.subject)
    models = String.(unique(panel.model))
    conds = length(conds) > max_curves ? conds[round.(Int, range(1, length(conds), length = max_curves))] : conds
    jobs = Tuple{Int,FitCondition,String,Vector{Float64},Float64}[]
    for (i, c) in enumerate(conds), m in models
        row = panel[(panel.subject .== c.name) .& (panel.model .== m), :]
        (nrow(row) == 0 || ismissing(row.params[1]) || isempty(string(row.params[1])) || !isfinite(row.wsse[1])) && continue
        s = sqrt(max(row.wsse[1], 1e-12) / row.n[1])
        push!(jobs, (i, c, m, parse_params(row.params[1]), s))
    end
    results = Vector{Any}(undef, length(jobs))
    done = Threads.Atomic{Int}(0)
    Threads.@threads :dynamic for j in eachindex(jobs)
        i, c, m, p, s = jobs[j]
        results[j] = cv_only ? (cv = [(; subject = c.name, model = m, showcase_cv(c, get_model(m); s2 = s^2)...)],) :
            check_curve(c, m, p, s; seed = i)
        Threads.atomic_add!(done, 1)
        done[] % 10 == 0 && @info "$(id): $(done[]) / $(length(jobs)) curve-model checks done"
    end
    for key in (cv_only ? (:cv,) : (:fisher, :profile, :boot, :recovery, :cv))
        CSV.write(joinpath(dir, "checks_$(key).csv"), DataFrame(reduce(vcat, [getfield(r, key) for r in results])))
    end
    return length(conds), length(models)
end

if abspath(PROGRAM_FILE) == @__FILE__
    k = findfirst(==("--only"), ARGS)
    ids = k === nothing ? String[] : [a for a in ARGS[(k + 1):end] if !startswith(a, "--") && tryparse(Int, a) === nothing]
    kc = findfirst(==("--curves"), ARGS)
    max_curves = kc === nothing ? 12 : parse(Int, ARGS[kc + 1])
    cfg = TOML.parsefile(joinpath(SHOWCASE, "pipeline.toml"))
    for d in cfg["dataset"]
        isempty(ids) || d["id"] in ids || continue
        c = merge(get(cfg, "defaults", Dict{String,Any}()), d)
        isfile(joinpath(RESULTS, c["id"], "meta.toml")) || continue
        t0 = time()
        nc, nm = package_checks(c["id"], dataset_conditions(c); max_curves, cv_only = "--cv-only" in ARGS)
        @info "$(c["id"]): package checks on $(nc) curves × $(nm) models in $(round(time() - t0; digits = 1)) s"
    end
end
