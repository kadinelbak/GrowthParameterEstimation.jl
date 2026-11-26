# The package's own diagnostics on the showcase picks: Fisher information, the built-in
# profile likelihood, residual bootstrap, synthetic recovery and k-fold cross-validation.
#
#   julia --project=showcase -t auto showcase/analyses/package_checks.jl --only <id> [--curves 12] [--cv-only | --profile-only]
#
# For each dataset, the rule pick is checked on an evenly spaced subset of curves, starting
# from the showcase fit. The weighted objective is reproduced exactly by fitting y/σ with an
# observable u -> u/σ(t), so every tool sees the same proportional error model as the pages.
# Writes results/<id>/checks_*.csv.

isdefined(@__MODULE__, :dataset_conditions) || include(joinpath(@__DIR__, "..", "run_pipeline.jl"))
using Random

const N_BOOT = 40
const N_RECOVERY = 20
const K_FOLDS = 5
const MAXITERS = 3000

parse_params(s) = parse.(Float64, split(string(s), ";"))

"""Wrap a registry model as the `(du, u, p, t)` function the joint-fit tools expect."""
function plain_rhs(spec, exposure)
    takes_exposure = applicable(spec.ode!, zeros(1), ones(1), ones(length(spec.param_names)), 0.0, exposure)
    return takes_exposure ? ((du, u, p, t) -> spec.ode!(du, u, p, t, exposure)) : ((du, u, p, t) -> spec.ode!(du, u, p, t))
end

"""One curve as a weighted joint-fit dataset: y/σ against u/σ(t), residual scale 1."""
function weighted_spec(c::FitCondition)
    sigma = Dict(zip(c.time, c.error))
    return [(x = c.time, y = c.count ./ c.error, observable = (u, p, t) -> u[1] / sigma[t], residual_scale = 1.0)]
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

function check_curve(c::FitCondition, model::String, p::Vector{Float64}; seed = 1, cv_only = false, profile_only = false,
                     stored_se = Dict{String,Float64}())
    spec = get_model(model)
    rhs = plain_rhs(spec, c.exposure)
    ds = weighted_spec(c)
    names = spec.param_names
    bounds = [(Float64(lo), Float64(hi)) for (lo, hi) in spec.bounds]
    rng = MersenneTwister(seed)
    out = (fisher = NamedTuple[], profile = NamedTuple[], boot = NamedTuple[], recovery = NamedTuple[], cv = NamedTuple[])

    cv_only && @goto cross_validation
    if profile_only   # reuse the stored Fisher errors
        fi = (table = (local_std_error = [stored_se[String(nm)] for nm in names],),)
        @goto profile
    end
    # 1. Fisher information at the showcase estimate (local, Wald-type intervals)
    fi = try
        fisher_information(rhs, ds, c.u0, p; parameter_names = names)
    catch
        nothing
    end
    for (j, nm) in enumerate(names)
        se = fi === nothing ? NaN : fi.table.local_std_error[j]
        push!(out.fisher, (subject = c.name, parameter = String(nm), estimate = p[j], std_error = se,
                           wald_lower = p[j] - 1.96se, wald_upper = p[j] + 1.96se,
                           full_rank = fi !== nothing && fi.numerical_rank == length(p),
                           condition_number = fi === nothing ? NaN : fi.condition_number))
    end

    @label profile
    # 2. Built-in profile likelihood, twice: with the package's default grid (8 log-spaced
    #    points per side spanning the whole bound range) and with that grid plus ±0.5…4 Fisher
    #    standard errors around the estimate (`fine_grid`).
    grid(j) = fine_grid(p[j], fi === nothing ? NaN : fi.table.local_std_error[j], bounds[j])
    profiles = Dict{String,Any}()
    for (label, vals) in (("default_grid", Dict{Symbol,Vector{Float64}}()),
                          ("fine_grid", Dict(names[j] => grid(j) for j in eachindex(names))))
        profiles[label] = try
            profile_likelihood(rhs, ds, c.u0, p; bounds, parameter_names = names, points_per_side = 8,
                               values = vals, optimizer = :nelder_mead, maxiters = MAXITERS)
        catch
            nothing
        end
    end
    for label in ("default_grid", "fine_grid"), (j, nm) in enumerate(names)
        pr = profiles[label]
        r = pr === nothing ? nothing : pr.confidence_intervals[j, :]
        g = [bounds[j][1], bounds[j][2]]   # both grids reach the bounds
        push!(out.profile, (subject = c.name, grid = label, parameter = String(nm), estimate = p[j],
                            lower = r === nothing ? NaN : r.lower, upper = r === nothing ? NaN : r.upper,
                            status = r === nothing ? "failed" : r.confidence_status,
                            grid_min = minimum(g), grid_max = maximum(g)))
    end

    profile_only && return out
    # 3. Residual bootstrap
    bs = try
        bootstrap_joint_fit(rhs, ds, c.u0, p; bounds, n_bootstrap = N_BOOT, method = :residual, rng,
                            optimizer = :nelder_mead, maxiters = MAXITERS)
    catch
        nothing
    end
    for (j, nm) in enumerate(names)
        push!(out.boot, (subject = c.name, parameter = String(nm), estimate = p[j],
                         lower = bs === nothing ? NaN : bs.summary.ci_lower[j],
                         upper = bs === nothing ? NaN : bs.summary.ci_upper[j],
                         success_rate = bs === nothing ? 0.0 : bs.success_rate))
    end

    # 4. Synthetic recovery: simulate from the estimate with the same error, refit
    rc = try
        synthetic_recovery_benchmark(rhs, ds, c.u0, p; bounds, n_simulations = N_RECOVERY, rng,
                                     optimizer = :nelder_mead, maxiters = MAXITERS)
    catch
        nothing
    end
    for (j, nm) in enumerate(names)
        push!(out.recovery, (subject = c.name, parameter = String(nm), truth = p[j],
                             rel_bias = rc === nothing ? NaN : rc.summary.bias[j] / abs(p[j]),
                             rel_rmse = rc === nothing ? NaN : rc.summary.rmse[j] / abs(p[j]),
                             success_rate = rc === nothing ? 0.0 : rc.success_rate))
    end

    # 5. k-fold cross-validation (points held out at random, unweighted)
    @label cross_validation
    Random.seed!(seed)
    cv = try
        redirect_stdout(devnull) do
            # explicit solver: the default Rodas5 differentiates in time, which fails on step exposures
            k_fold_cross_validation(c.time, c.count, p; k_folds = min(K_FOLDS, length(c.time) ÷ 2),
                                    model = rhs, bounds = bounds, solver = GrowthParameterEstimation.Analysis.Tsit5())
        end
    catch
        nothing
    end
    push!(out.cv, (subject = c.name, r_squared = cv === nothing ? NaN : cv.r_squared,
                   rel_rmse = cv === nothing ? NaN : cv.overall_rmse / mean(c.count)))
    return out
end

function package_checks(id::AbstractString, conds::Vector{FitCondition}; max_curves = 12, cv_only = false, profile_only = false)
    dir = joinpath(RESULTS, id)
    meta = TOML.parsefile(joinpath(dir, "meta.toml"))
    model = meta["pick"]
    panel = CSV.read(joinpath(dir, "panel.csv"), DataFrame)
    conds = length(conds) > max_curves ? conds[round.(Int, range(1, length(conds), length = max_curves))] : conds
    results = Vector{Any}(undef, length(conds))
    Threads.@threads for i in eachindex(conds)
        c = conds[i]
        row = panel[(string.(panel.subject) .== c.name) .& (panel.model .== model), :]
        if nrow(row) == 0 || ismissing(row.params[1]) || isempty(string(row.params[1]))
            results[i] = nothing
            continue
        end
        se = Dict{String,Float64}()
        if profile_only
            f = CSV.read(joinpath(dir, "checks_fisher.csv"), DataFrame)
            f = f[string.(f.subject) .== c.name, :]
            se = Dict(String(r.parameter) => Float64(r.std_error) for r in eachrow(f))
        end
        results[i] = check_curve(c, model, parse_params(row.params[1]); seed = i, cv_only, profile_only, stored_se = se)
    end
    ok = [r for r in results if r !== nothing]
    for key in (cv_only ? (:cv,) : profile_only ? (:profile,) : (:fisher, :profile, :boot, :recovery, :cv))
        CSV.write(joinpath(dir, "checks_$(key).csv"), DataFrame(reduce(vcat, [getfield(r, key) for r in ok])))
    end
    return length(ok)
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
        conds = dataset_conditions(c)
        n = package_checks(c["id"], conds; max_curves, cv_only = "--cv-only" in ARGS,
                           profile_only = "--profile-only" in ARGS)
        @info "$(c["id"]): package checks on $(n) curves in $(round(time() - t0; digits = 1)) s"
    end
end
