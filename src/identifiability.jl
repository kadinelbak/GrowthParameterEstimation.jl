"""
Practical identifiability tools for joint ODE fits.

The numerical diagnostics in this module are deliberately separated from
structural identifiability. A full structural result requires a symbolic
differential-algebra backend and an explicit observation map; this module
therefore reports that requirement instead of inferring structural
identifiability from a successful optimizer run.
"""
module Identifiability

using DataFrames
using DifferentialEquations
using LinearAlgebra
using Random
using StructuralIdentifiability
using Statistics

using ..Fitting

export ObservationMap, IdentifiabilityConfig,
    validate_observation_map, generate_multistarts,
    prediction_vector, prediction_jacobian, fisher_information,
    profile_likelihood, bootstrap_joint_fit, synthetic_recovery_benchmark,
    practical_identifiability_report, structural_identifiability,
    structural_identifiability_report

"""
    ObservationMap(name, state_labels, observed_labels; description="")

Document the link between a model's state vector and the quantities measured
by an assay. `observed_labels` should have one entry per `dataset_specs` item
passed to `run_joint_fit`; an item can be a state index or an arbitrary
observable function.
"""
struct ObservationMap
    name::String
    state_labels::Vector{Symbol}
    observed_labels::Vector{Symbol}
    description::String
end

function ObservationMap(
    name::AbstractString,
    state_labels::Vector{Symbol},
    observed_labels::Vector{Symbol};
    description::AbstractString = "",
)
    isempty(state_labels) && throw(ArgumentError("state_labels cannot be empty"))
    isempty(observed_labels) && throw(ArgumentError("observed_labels cannot be empty"))
    return ObservationMap(String(name), state_labels, observed_labels, String(description))
end

"""
    IdentifiabilityConfig(; parameter_names, bounds, ...)

Settings shared by the practical-identifiability routines. Bounds are part of
the scientific model: a parameter constrained by a narrow prior range should
not be described as data-identifiable without reporting that restriction.
"""
Base.@kwdef struct IdentifiabilityConfig
    parameter_names::Vector{Symbol}
    bounds::Vector{Tuple{Float64,Float64}}
    n_starts::Int = 20
    start_scale::Symbol = :log
    multistart_bic_tolerance::Float64 = 2.0
    cluster_tolerance::Float64 = 0.05
    finite_difference_step::Float64 = 1e-5
    rank_tolerance::Float64 = 1e-8
    profile_points_per_side::Int = 8
    profile_threshold::Float64 = 3.841458820694124
    bootstrap_replicates::Int = 0
    bootstrap_method::Symbol = :parametric
    bootstrap_success_threshold::Float64 = 0.8
end

function IdentifiabilityConfig(
    parameter_names::Vector{Symbol},
    bounds::Vector{<:Tuple};
    kwargs...,
)
    length(parameter_names) == length(bounds) ||
        throw(ArgumentError("parameter_names and bounds must have equal length"))
    normalized_bounds = [(Float64(lo), Float64(hi)) for (lo, hi) in bounds]
    all(lo < hi for (lo, hi) in normalized_bounds) ||
        throw(ArgumentError("each parameter bound must have lower < upper"))
    return IdentifiabilityConfig(; parameter_names = parameter_names, bounds = normalized_bounds, kwargs...)
end

"""
    validate_observation_map(map, dataset_specs)

Validate that a documented `ObservationMap` matches the datasets supplied to a
joint fit. The return value records direct state measurements versus custom
observable functions, because that distinction changes structural
identifiability.
"""
function validate_observation_map(
    map::ObservationMap,
    dataset_specs::Vector{<:NamedTuple},
)
    length(map.observed_labels) == length(dataset_specs) ||
        throw(ArgumentError("ObservationMap has $(length(map.observed_labels)) labels for $(length(dataset_specs)) datasets"))

    rows = NamedTuple[]
    for (index, ds) in enumerate(dataset_specs)
        direct_state = haskey(ds, :state_index)
        custom_observable = haskey(ds, :observable)
        (direct_state || custom_observable) ||
            throw(ArgumentError("dataset $index requires state_index or observable"))
        if direct_state && !(1 <= Int(ds.state_index) <= length(map.state_labels))
            throw(ArgumentError("dataset $index state_index does not exist in ObservationMap"))
        end
        push!(rows, (
            dataset = index,
            label = String(map.observed_labels[index]),
            measurement = direct_state ? "state" : "custom_observable",
            mapped_state = direct_state ? String(map.state_labels[Int(ds.state_index)]) : "",
            n_observations = length(ds.y),
        ))
    end
    return DataFrame(rows)
end

function _processed_datasets(dataset_specs::Vector{<:NamedTuple})
    isempty(dataset_specs) && throw(ArgumentError("dataset_specs cannot be empty"))
    processed = NamedTuple[]
    for ds in dataset_specs
        haskey(ds, :x) && haskey(ds, :y) ||
            throw(ArgumentError("each dataset requires x and y"))
        x = Float64.(collect(ds.x))
        y = Float64.(collect(ds.y))
        length(x) == length(y) || throw(ArgumentError("x and y length mismatch"))
        push!(processed, (
            x = x,
            y = y,
            state_index = haskey(ds, :state_index) ? Int(ds.state_index) : 0,
            observable = haskey(ds, :observable) ? ds.observable : nothing,
            residual_scale = haskey(ds, :residual_scale) ? max(Float64(ds.residual_scale), eps(Float64)) : 1.0,
        ))
    end
    return processed
end

function _observable(ds, state, p, t)
    ds.observable === nothing && return state[ds.state_index]
    if applicable(ds.observable, state, p, t)
        return ds.observable(state, p, t)
    elseif applicable(ds.observable, state, p)
        return ds.observable(state, p)
    elseif applicable(ds.observable, state)
        return ds.observable(state)
    end
    throw(ArgumentError("observable must accept (u, p, t), (u, p), or u"))
end

function _initial_state(u0, u0_builder, p)
    values = u0_builder === nothing ? u0 : u0_builder(p)
    length(values) == length(u0) || throw(ArgumentError("u0_builder returned the wrong number of states"))
    return Float64.(collect(values))
end

"""
    prediction_vector(model, dataset_specs, u0, p; ...)

Return concatenated predictions in the same dataset/time order used by
`run_joint_fit`. Set `weighted=true` to divide each dataset by its declared
`residual_scale`, which is the form used for Fisher-information calculations.
"""
function prediction_vector(
    model::Function,
    dataset_specs::Vector{<:NamedTuple},
    u0::Vector{<:Real},
    p::Vector{<:Real};
    solver = Tsit5(),
    u0_builder = nothing,
    initial_time = nothing,
    reltol::Real = 1e-10,
    abstol::Real = 1e-10,
    weighted::Bool = false,
)
    processed = _processed_datasets(dataset_specs)
    save_times = sort(unique(vcat([ds.x for ds in processed]...)))
    t0 = initial_time === nothing ? first(save_times) : Float64(initial_time)
    t0 <= first(save_times) || throw(ArgumentError("initial_time must be no later than the first observation"))
    p_float = Float64.(p)
    prob = ODEProblem(model, _initial_state(u0, u0_builder, p_float), (t0, last(save_times)), p_float)
    sol = solve(prob, solver; saveat = save_times, reltol = reltol, abstol = abstol)
    sol.retcode == ReturnCode.Success || error("ODE solve failed while evaluating predictions")

    values = Float64[]
    for ds in processed
        for time in ds.x
            index = findfirst(t -> isapprox(t, time; atol = 1e-10, rtol = 1e-10), save_times)
            index === nothing && error("prediction time not found in solution")
            value = Float64(_observable(ds, sol.u[index], p_float, time))
            isfinite(value) || error("model produced a non-finite observation")
            push!(values, weighted ? value / ds.residual_scale : value)
        end
    end
    return values
end

function _observation_vector(dataset_specs::Vector{<:NamedTuple}; weighted::Bool)
    processed = _processed_datasets(dataset_specs)
    return reduce(vcat, [weighted ? ds.y ./ ds.residual_scale : ds.y for ds in processed])
end

"""
    generate_multistarts(bounds; n_starts=20, scale=:log, rng=Random.default_rng())

Generate reproducible bounded starts. `scale=:log` samples positive parameters
uniformly in log space and falls back to linear sampling for intervals touching
zero or crossing zero.
"""
function generate_multistarts(
    bounds::Vector{<:Tuple};
    n_starts::Int = 20,
    scale::Symbol = :log,
    rng::AbstractRNG = Random.default_rng(),
)
    n_starts > 0 || throw(ArgumentError("n_starts must be positive"))
    scale in (:log, :linear) || throw(ArgumentError("scale must be :log or :linear"))
    starts = Vector{Vector{Float64}}(undef, n_starts)
    for start_index in 1:n_starts
        starts[start_index] = [
            if scale == :log && lo > 0 && hi > 0
                exp(log(Float64(lo)) + rand(rng) * (log(Float64(hi)) - log(Float64(lo))))
            else
                Float64(lo) + rand(rng) * (Float64(hi) - Float64(lo))
            end
            for (lo, hi) in bounds
        ]
    end
    return starts
end

"""
    prediction_jacobian(model, dataset_specs, u0, p; ...)

Central finite-difference Jacobian of weighted model predictions with respect
to parameters. The output is robust to arbitrary Julia ODE right-hand sides;
for parameters whose scale matters, use `log_parameters=true`.
"""
function prediction_jacobian(
    model::Function,
    dataset_specs::Vector{<:NamedTuple},
    u0::Vector{<:Real},
    p::Vector{<:Real};
    finite_difference_step::Real = 1e-5,
    log_parameters::Bool = true,
    kwargs...,
)
    p_float = Float64.(p)
    base = prediction_vector(model, dataset_specs, u0, p_float; weighted = true, kwargs...)
    jacobian = Matrix{Float64}(undef, length(base), length(p_float))
    steps = zeros(Float64, length(p_float))

    for index in eachindex(p_float)
        step = if log_parameters && p_float[index] > 0
            max(abs(p_float[index]) * Float64(finite_difference_step), eps(Float64))
        else
            max(abs(p_float[index]), 1.0) * Float64(finite_difference_step)
        end
        lower = copy(p_float)
        upper = copy(p_float)
        lower[index] -= step
        upper[index] += step
        if log_parameters && p_float[index] > 0 && lower[index] <= 0
            lower[index] = p_float[index]
            upper[index] = p_float[index] + step
            jacobian[:, index] .= (prediction_vector(model, dataset_specs, u0, upper; weighted = true, kwargs...) .- base) ./ step
        else
            jacobian[:, index] .= (
                prediction_vector(model, dataset_specs, u0, upper; weighted = true, kwargs...) .-
                prediction_vector(model, dataset_specs, u0, lower; weighted = true, kwargs...)
            ) ./ (2 * step)
        end
        steps[index] = step
    end
    return (jacobian = jacobian, baseline = base, steps = steps, parameterization = log_parameters ? :parameter_scale : :linear)
end

"""
    fisher_information(model, dataset_specs, u0, p; parameter_names, ...)

Compute a weighted prediction Jacobian, Fisher information `J'J`, numerical
rank, singular values, and the local covariance/correlation approximation.
This is a *local practical* diagnostic, not a proof of structural
identifiability.
"""
function fisher_information(
    model::Function,
    dataset_specs::Vector{<:NamedTuple},
    u0::Vector{<:Real},
    p::Vector{<:Real};
    parameter_names::Vector{Symbol} = Symbol.("p" .* string.(collect(eachindex(p)))),
    rank_tolerance::Real = 1e-8,
    kwargs...,
)
    length(parameter_names) == length(p) || throw(ArgumentError("parameter_names must match p"))
    derivative = prediction_jacobian(model, dataset_specs, u0, p; kwargs...)
    jacobian = derivative.jacobian
    information = jacobian' * jacobian
    singular_values = svdvals(jacobian)
    maximum_singular = isempty(singular_values) ? 0.0 : maximum(singular_values)
    threshold = max(Float64(rank_tolerance) * maximum_singular, eps(Float64))
    numerical_rank = count(>=(threshold), singular_values)
    condition_number = numerical_rank == length(p) && !isempty(singular_values) ?
        maximum_singular / max(minimum(singular_values), eps(Float64)) : Inf
    covariance = pinv(information; rtol = Float64(rank_tolerance))
    std_errors = sqrt.(max.(diag(covariance), 0.0))
    correlation = Matrix{Float64}(undef, length(p), length(p))
    for i in eachindex(p), j in eachindex(p)
        denominator = std_errors[i] * std_errors[j]
        correlation[i, j] = denominator > eps(Float64) ? covariance[i, j] / denominator : NaN
    end
    status = numerical_rank == length(p) ? "full_rank_local" : "rank_deficient_local"
    table = DataFrame(
        parameter = String.(parameter_names),
        estimate = Float64.(p),
        local_std_error = std_errors,
        local_identifiability = fill(status, length(p)),
    )
    return (
        table = table,
        jacobian = jacobian,
        fisher_information = information,
        covariance = covariance,
        correlation = correlation,
        singular_values = singular_values,
        numerical_rank = numerical_rank,
        condition_number = condition_number,
        status = status,
        finite_difference_steps = derivative.steps,
    )
end

function _profile_grid(estimate::Float64, lower::Float64, upper::Float64, points_per_side::Int)
    points_per_side >= 1 || throw(ArgumentError("points_per_side must be positive"))
    left = if lower < estimate
        lower > 0 && estimate > 0 ? exp.(range(log(lower), log(estimate), length = points_per_side + 1))[1:end-1] :
            collect(range(lower, estimate, length = points_per_side + 1))[1:end-1]
    else
        Float64[]
    end
    right = if estimate < upper
        estimate > 0 && upper > 0 ? exp.(range(log(estimate), log(upper), length = points_per_side + 1))[2:end] :
            collect(range(estimate, upper, length = points_per_side + 1))[2:end]
    else
        Float64[]
    end
    return sort(unique(vcat(left, [estimate], right)))
end

function _profiled_dataset_specs(dataset_specs, parameter_index, fixed_value)
    transformed = NamedTuple[]
    full_params = function (free)
        p = Vector{Float64}(undef, length(free) + 1)
        p[parameter_index] = fixed_value
        free_index = 1
        for index in eachindex(p)
            index == parameter_index && continue
            p[index] = free[free_index]
            free_index += 1
        end
        return p
    end
    for ds in dataset_specs
        if haskey(ds, :observable)
            observation = ds.observable
            wrapped = (u, free, t) -> _observable((observable = observation,), u, full_params(free), t)
            push!(transformed, merge(ds, (observable = wrapped,)))
        else
            push!(transformed, ds)
        end
    end
    return transformed, full_params
end

"""
    profile_likelihood(model, dataset_specs, u0, p0; bounds, parameter_names, ...)

Fix each parameter over a grid, re-optimize all remaining parameters, and
report the weighted-SSE profile. `confidence_status` is based on the supplied
`threshold` (default: one-parameter 95% chi-square cutoff) and is meaningful
only when `residual_scale` represents a known or defensible measurement SD.
"""
function profile_likelihood(
    model::Function,
    dataset_specs::Vector{<:NamedTuple},
    u0::Vector{<:Real},
    p0::Vector{<:Real};
    bounds::Vector{<:Tuple},
    parameter_names::Vector{Symbol},
    values::AbstractDict = Dict{Symbol,Vector{Float64}}(),
    points_per_side::Int = 8,
    threshold::Real = 3.841458820694124,
    solver = Tsit5(),
    u0_builder = nothing,
    initial_time = nothing,
    optimizer::Symbol = :nelder_mead,
    maxiters::Integer = 10_000,
    show_stats::Bool = false,
)
    length(p0) == length(bounds) == length(parameter_names) ||
        throw(ArgumentError("p0, bounds, and parameter_names must have equal length"))
    baseline = Fitting.run_joint_fit(
        model, dataset_specs, u0, Float64.(p0);
        solver = solver, bounds = bounds, u0_builder = u0_builder,
        initial_time = initial_time, optimizer = optimizer, maxiters = maxiters,
        show_stats = show_stats,
    )
    rows = NamedTuple[]
    confidence_rows = NamedTuple[]
    for index in eachindex(parameter_names)
        name = parameter_names[index]
        grid = haskey(values, name) ? sort(unique(Float64.(values[name]))) :
            _profile_grid(Float64(baseline.params[index]), Float64(bounds[index][1]), Float64(bounds[index][2]), points_per_side)
        grid = filter(value -> bounds[index][1] <= value <= bounds[index][2], grid)
        isempty(grid) && throw(ArgumentError("profile grid for $name has no values inside its bounds"))
        parameter_rows = NamedTuple[]
        for value in grid
            profiled_specs, expand_params = _profiled_dataset_specs(dataset_specs, index, value)
            free_start = [baseline.params[j] for j in eachindex(p0) if j != index]
            free_bounds = [bounds[j] for j in eachindex(bounds) if j != index]
            free_model! = (du, u, free, t) -> model(du, u, expand_params(free), t)
            free_u0_builder = u0_builder === nothing ? nothing : free -> u0_builder(expand_params(free))
            fit = try
                Fitting.run_joint_fit(
                    free_model!, profiled_specs, u0, free_start;
                    solver = solver, bounds = free_bounds, u0_builder = free_u0_builder,
                    initial_time = initial_time, optimizer = optimizer, maxiters = maxiters,
                    show_stats = false,
                )
            catch
                nothing
            end
            sse = fit === nothing ? Inf : Float64(fit.sse)
            params = fit === nothing ? fill(NaN, length(p0)) : expand_params(fit.params)
            row = (
                parameter = String(name),
                parameter_index = index,
                fixed_value = value,
                sse = sse,
                delta_sse = sse - Float64(baseline.sse),
                accepted = isfinite(sse) && sse - Float64(baseline.sse) <= threshold,
                params = string(params),
            )
            push!(rows, row)
            push!(parameter_rows, row)
        end
        accepted_values = [row.fixed_value for row in parameter_rows if row.accepted]
        lower_closed = !isempty(accepted_values) && minimum(accepted_values) > Float64(bounds[index][1])
        upper_closed = !isempty(accepted_values) && maximum(accepted_values) < Float64(bounds[index][2])
        push!(confidence_rows, (
            parameter = String(name),
            estimate = Float64(baseline.params[index]),
            lower = isempty(accepted_values) ? NaN : minimum(accepted_values),
            upper = isempty(accepted_values) ? NaN : maximum(accepted_values),
            confidence_status = lower_closed && upper_closed ? "bounded" :
                lower_closed ? "upper_unbounded" : upper_closed ? "lower_unbounded" : "unbounded_or_flat",
        ))
    end
    return (fit = baseline, profile = DataFrame(rows), confidence_intervals = DataFrame(confidence_rows), threshold = Float64(threshold))
end

function _cluster_multistarts(summary::DataFrame, bounds::Vector{<:Tuple}; bic_tolerance::Real, cluster_tolerance::Real)
    completed = filter(row -> row.status == "completed" && isfinite(row.bic), eachrow(summary))
    isempty(completed) && return DataFrame(cluster = Int[], n_starts = Int[], best_bic = Float64[], representative_params = String[])
    best_bic = minimum(row.bic for row in completed)
    candidates = [row for row in completed if row.bic <= best_bic + bic_tolerance]
    representatives = Vector{Vector{Float64}}()
    counts = Int[]
    cluster_bics = Float64[]
    for row in candidates
        params = try
            parse.(Float64, split(replace(replace(row.params, '[' => ""), ']' => ""), ','))
        catch
            Float64[]
        end
        length(params) == length(bounds) || continue
        normalized = [(params[i] - bounds[i][1]) / (bounds[i][2] - bounds[i][1]) for i in eachindex(params)]
        matched = findfirst(reference -> norm(normalized .- reference) <= cluster_tolerance, representatives)
        if matched === nothing
            push!(representatives, normalized)
            push!(counts, 1)
            push!(cluster_bics, row.bic)
        else
            counts[matched] += 1
            cluster_bics[matched] = min(cluster_bics[matched], row.bic)
        end
    end
    return DataFrame(
        cluster = collect(eachindex(representatives)),
        n_starts = counts,
        best_bic = cluster_bics,
        representative_params = [string(representative) for representative in representatives],
    )
end

"""
    bootstrap_joint_fit(model, dataset_specs, u0, p0; ...)

Refit residual or parametric bootstrap data sets. `method=:residual` resamples
within each dataset; `method=:parametric` draws Gaussian noise using each
dataset's `residual_scale`. Treat each dataset as a distinct assay/replicate
stratum when constructing `dataset_specs`.
"""
function bootstrap_joint_fit(
    model::Function,
    dataset_specs::Vector{<:NamedTuple},
    u0::Vector{<:Real},
    p0::Vector{<:Real};
    bounds::Vector{<:Tuple},
    n_bootstrap::Int = 100,
    method::Symbol = :residual,
    rng::AbstractRNG = Random.default_rng(),
    kwargs...,
)
    n_bootstrap > 0 || throw(ArgumentError("n_bootstrap must be positive"))
    method in (:residual, :parametric) || throw(ArgumentError("method must be :residual or :parametric"))
    baseline = Fitting.run_joint_fit(model, dataset_specs, u0, Float64.(p0); bounds = bounds, kwargs...)
    processed = _processed_datasets(dataset_specs)
    predictions = baseline.predictions
    rows = NamedTuple[]
    estimates = Vector{Vector{Float64}}()
    for replicate in 1:n_bootstrap
        boot_specs = NamedTuple[]
        for (index, ds) in enumerate(processed)
            residuals = ds.y .- predictions[index]
            noise = method == :residual ? rand(rng, residuals, length(residuals)) : ds.residual_scale .* randn(rng, length(residuals))
            original = dataset_specs[index]
            push!(boot_specs, merge(original, (y = predictions[index] .+ noise,)))
        end
        fit = try
            Fitting.run_joint_fit(model, boot_specs, u0, baseline.params; bounds = bounds, kwargs...)
        catch
            nothing
        end
        if fit === nothing || !isfinite(fit.sse) || fit.sse >= 9.99e11
            push!(rows, (replicate = replicate, status = "failed", sse = NaN, params = ""))
        else
            push!(estimates, Float64.(fit.params))
            push!(rows, (replicate = replicate, status = "completed", sse = Float64(fit.sse), params = string(Float64.(fit.params))))
        end
    end
    parameter_count = length(p0)
    summary = DataFrame(
        parameter_index = collect(eachindex(p0)),
        estimate = baseline.params,
        bootstrap_mean = [isempty(estimates) ? NaN : mean(p[index] for p in estimates) for index in 1:parameter_count],
        bootstrap_std = [length(estimates) < 2 ? NaN : std(p[index] for p in estimates) for index in 1:parameter_count],
        ci_lower = [isempty(estimates) ? NaN : quantile([p[index] for p in estimates], 0.025) for index in 1:parameter_count],
        ci_upper = [isempty(estimates) ? NaN : quantile([p[index] for p in estimates], 0.975) for index in 1:parameter_count],
    )
    return (fit = baseline, replicates = DataFrame(rows), summary = summary, estimates = estimates, success_rate = length(estimates) / n_bootstrap)
end

"""
    synthetic_recovery_benchmark(model, dataset_specs, u0, p_true; ...)

Simulate noisy observations from `p_true`, refit them, and summarize parameter
recovery. This is the primary pre-real-data check that a proposed experiment
can estimate its intended parameters.
"""
function synthetic_recovery_benchmark(
    model::Function,
    dataset_specs::Vector{<:NamedTuple},
    u0::Vector{<:Real},
    p_true::Vector{<:Real};
    bounds::Vector{<:Tuple},
    starts::Union{Nothing,Vector{<:AbstractVector}} = nothing,
    n_simulations::Int = 100,
    rng::AbstractRNG = Random.default_rng(),
    solver = Tsit5(),
    u0_builder = nothing,
    initial_time = nothing,
    optimizer::Symbol = :nelder_mead,
    maxiters::Integer = 10_000,
)
    n_simulations > 0 || throw(ArgumentError("n_simulations must be positive"))
    truth = prediction_vector(model, dataset_specs, u0, p_true;
        solver = solver, u0_builder = u0_builder, initial_time = initial_time)
    processed = _processed_datasets(dataset_specs)
    splits = cumsum(length(ds.y) for ds in processed)
    rows = NamedTuple[]
    estimates = Vector{Vector{Float64}}()
    active_starts = starts === nothing ? [Float64.(p_true)] : starts
    for simulation in 1:n_simulations
        noisy_specs = NamedTuple[]
        first_index = 1
        for (dataset_index, ds) in enumerate(processed)
            last_index = splits[dataset_index]
            y = truth[first_index:last_index] .+ ds.residual_scale .* randn(rng, length(ds.y))
            push!(noisy_specs, merge(dataset_specs[dataset_index], (y = y,)))
            first_index = last_index + 1
        end
        fitted = try
            Fitting.run_joint_multistart(
                model, noisy_specs, u0, active_starts;
                bounds = bounds, solver = solver, u0_builder = u0_builder,
                initial_time = initial_time, optimizer = optimizer, maxiters = maxiters,
            ).fit
        catch
            nothing
        end
        if fitted === nothing
            push!(rows, (simulation = simulation, status = "failed", params = ""))
        else
            params = Float64.(fitted.params)
            push!(estimates, params)
            push!(rows, (simulation = simulation, status = "completed", params = string(params)))
        end
    end
    parameter_count = length(p_true)
    summary = DataFrame(
        parameter_index = collect(eachindex(p_true)),
        truth = Float64.(p_true),
        mean_estimate = [isempty(estimates) ? NaN : mean(p[index] for p in estimates) for index in 1:parameter_count],
        bias = [isempty(estimates) ? NaN : mean(p[index] for p in estimates) - p_true[index] for index in 1:parameter_count],
        rmse = [isempty(estimates) ? NaN : sqrt(mean((p[index] - p_true[index])^2 for p in estimates)) for index in 1:parameter_count],
    )
    return (replicates = DataFrame(rows), summary = summary, estimates = estimates, success_rate = length(estimates) / n_simulations)
end

"""
    practical_identifiability_report(model, dataset_specs, u0, p0; config, ...)

Run a reproducible multistart fit, local Fisher-information diagnostic, and
profile likelihood. The returned `status` is deliberately conservative: a
full-rank FIM is necessary but not sufficient for a practically identifiable
parameterization.
"""
function practical_identifiability_report(
    model::Function,
    dataset_specs::Vector{<:NamedTuple},
    u0::Vector{<:Real},
    p0::Vector{<:Real};
    config::IdentifiabilityConfig,
    starts::Union{Nothing,Vector{<:AbstractVector}} = nothing,
    rng::AbstractRNG = Random.default_rng(),
    solver = Tsit5(),
    u0_builder = nothing,
    initial_time = nothing,
    optimizer::Symbol = :nelder_mead,
    maxiters::Integer = 10_000,
    reltol::Real = 1e-10,
    abstol::Real = 1e-10,
)
    length(p0) == length(config.parameter_names) || throw(ArgumentError("p0 must match config.parameter_names"))
    active_starts = starts === nothing ? generate_multistarts(config.bounds; n_starts = config.n_starts, scale = config.start_scale, rng = rng) : starts
    multistart = Fitting.run_joint_multistart(
        model, dataset_specs, u0, active_starts;
        bounds = config.bounds, solver = solver, u0_builder = u0_builder,
        initial_time = initial_time, optimizer = optimizer, maxiters = maxiters,
        reltol = reltol, abstol = abstol,
    )
    clusters = _cluster_multistarts(multistart.summary, config.bounds;
        bic_tolerance = config.multistart_bic_tolerance, cluster_tolerance = config.cluster_tolerance)
    fisher = fisher_information(model, dataset_specs, u0, multistart.fit.params;
        parameter_names = config.parameter_names, rank_tolerance = config.rank_tolerance,
        finite_difference_step = config.finite_difference_step, solver = solver,
        u0_builder = u0_builder, initial_time = initial_time, reltol = reltol, abstol = abstol)
    profiles = profile_likelihood(model, dataset_specs, u0, multistart.fit.params;
        bounds = config.bounds, parameter_names = config.parameter_names,
        points_per_side = config.profile_points_per_side, threshold = config.profile_threshold,
        solver = solver, u0_builder = u0_builder, initial_time = initial_time,
        optimizer = optimizer, maxiters = maxiters)
    bootstrap = config.bootstrap_replicates > 0 ? bootstrap_joint_fit(
        model, dataset_specs, u0, multistart.fit.params;
        bounds = config.bounds, n_bootstrap = config.bootstrap_replicates,
        method = config.bootstrap_method, rng = rng, solver = solver,
        u0_builder = u0_builder, initial_time = initial_time, optimizer = optimizer,
        maxiters = maxiters, reltol = reltol, abstol = abstol,
    ) : nothing
    profile_ok = all(profiles.confidence_intervals.confidence_status .== "bounded")
    bootstrap_ok = bootstrap === nothing || bootstrap.success_rate >= config.bootstrap_success_threshold
    status = fisher.numerical_rank == length(p0) && nrow(clusters) == 1 && profile_ok && bootstrap_ok ? "passes_numerical_gates" : "requires_review"
    return (
        status = status,
        fit = multistart.fit,
        multistart = multistart,
        solution_clusters = clusters,
        fisher = fisher,
        profiles = profiles,
        bootstrap = bootstrap,
        observation_count = length(_observation_vector(dataset_specs; weighted = false)),
    )
end

"""
    structural_identifiability(ode; mode=:global, funcs_to_check=Any[], ...)

Run symbolic structural identifiability using `StructuralIdentifiability.jl`.
`ode` must be a symbolic `@ODEmodel` from that package, with output equations
that exactly match the documented `ObservationMap`. `mode=:global` returns the
three-way `:globally`, `:locally`, or `:nonidentifiable` classification;
`mode=:local` runs the faster local test and returns `:locally` or
`:nonidentifiable`.
"""
function structural_identifiability(
    ode;
    mode::Symbol = :global,
    funcs_to_check::Vector = Any[],
    known_ic::Vector = Any[],
    prob_threshold::Real = 0.99,
    experiment_type::Symbol = :SE,
)
    0 < prob_threshold < 1 || throw(ArgumentError("prob_threshold must be between zero and one"))
    mode in (:global, :local) || throw(ArgumentError("mode must be :global or :local"))
    experiment_type in (:SE, :ME) || throw(ArgumentError("experiment_type must be :SE or :ME"))

    raw = if mode == :global
        isempty(known_ic) ?
            StructuralIdentifiability.assess_identifiability(
                ode;
                funcs_to_check = funcs_to_check,
                prob_threshold = Float64(prob_threshold),
            ) :
            StructuralIdentifiability.assess_identifiability(
                ode;
                funcs_to_check = funcs_to_check,
                known_ic = known_ic,
                prob_threshold = Float64(prob_threshold),
            )
    else
        isempty(known_ic) || throw(ArgumentError("known_ic is currently supported only for mode=:global"))
        StructuralIdentifiability.assess_local_identifiability(
            ode;
            funcs_to_check = funcs_to_check,
            prob_threshold = Float64(prob_threshold),
            type = experiment_type,
        )
    end
    statuses = mode == :global ? Symbol.(collect(values(raw))) :
        [value ? :locally : :nonidentifiable for value in values(raw)]
    table = DataFrame(
        term = string.(collect(keys(raw))),
        identifiability = String.(statuses),
        mode = fill(String(mode), length(statuses)),
    )
    return (
        table = table,
        raw = raw,
        mode = mode,
        prob_threshold = Float64(prob_threshold),
        experiment_type = experiment_type,
    )
end

"""
    structural_identifiability_report(map, parameter_names; backend=nothing, known_inputs=Symbol[])

Create the required structural-identifiability record. Supply `backend` as a
function accepting `(map, parameter_names, known_inputs)` and returning a
table or named tuple from a symbolic differential-algebra analysis. Without a
backend this returns `requires_symbolic_backend`; it never mislabels numerical
fit stability as a global or local structural result.
"""
function structural_identifiability_report(
    map::ObservationMap,
    parameter_names::Vector{Symbol};
    known_inputs::Vector{Symbol} = Symbol[],
    backend = nothing,
)
    if backend === nothing
        return (
            status = "requires_symbolic_backend",
            model = map.name,
            parameters = parameter_names,
            observed_states = map.observed_labels,
            known_inputs = known_inputs,
            recommendation = "Run a symbolic differential-algebra backend with this exact observation map before interpreting global or local structural identifiability.",
        )
    end
    result = backend(map, parameter_names, known_inputs)
    return (
        status = "backend_completed",
        model = map.name,
        parameters = parameter_names,
        observed_states = map.observed_labels,
        known_inputs = known_inputs,
        result = result,
    )
end

end # module Identifiability
