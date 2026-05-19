using GenieFramework
using StippleUI
using StipplePlotly
using CSV
using DataFrames
using GrowthParameterEstimation
using Statistics
using Base64
using TOML
using Random

@genietools

const HOST = "127.0.0.1"
const PORT = parse(Int, get(ENV, "GPE_GUI_PORT", "8050"))
const LATEX_CONFIG_PATH = joinpath(dirname(dirname(dirname(@__FILE__))), "config", "model_latex.toml")
const EXAMPLE_DIR = joinpath(dirname(@__FILE__), "data")
const GUI_CUSTOM_MODELS_PATH = joinpath(EXAMPLE_DIR, "gui_custom_models.toml")
const GUI_PIPELINES_PATH = joinpath(EXAMPLE_DIR, "gui_pipelines.toml")
const SAFE_LOAD_CACHE = Dict{String,Tuple{String,DataFrame}}()

# ── Data structures ───────────────────────────────────────────────────────────
mutable struct PipelineStage
    name::String
    csv_file::String
    model_name::String
    param_mapping::Dict{String, String}
end

mutable struct Pipeline
    name::String
    stages::Vector{PipelineStage}
end

# ── LaTeX cache ───────────────────────────────────────────────────────────────
const LATEX_CACHE = Dict{String, String}()
let
    if isfile(LATEX_CONFIG_PATH)
        try
            config = TOML.parsefile(LATEX_CONFIG_PATH)
            if haskey(config, "latex")
                for (model, latex) in config["latex"]
                    LATEX_CACHE[model] = latex
                end
            end
        catch
        end
    end
end

# ── Model catalog ─────────────────────────────────────────────────────────────
const FAMILY_SORT_ORDER = ["logistic", "gompertz", "theta_logistic", "coculture", "mechanistic", "custom", "legacy"]
const FAMILY_LABELS = Dict(
    "logistic"       => "Logistic Family (baseline)",
    "gompertz"       => "Gompertz Family (baseline)",
    "theta_logistic" => "Theta-Logistic + Drug Response",
    "coculture"      => "Co-culture / Competition",
    "mechanistic"    => "Mechanistic (sensitive-resistant)",
    "custom"         => "Custom Models",
    "legacy"         => "Other",
)

function _spec_family(spec)
    if hasproperty(spec, :base_growth_family)
        fam = strip(String(getproperty(spec, :base_growth_family)))
        return isempty(fam) ? "legacy" : fam
    end
    if hasproperty(spec, :metadata)
        md = getproperty(spec, :metadata)
        if md isa AbstractDict
            if haskey(md, :family)
                fam = strip(String(md[:family]))
                return isempty(fam) ? "legacy" : fam
            elseif haskey(md, "family")
                fam = strip(String(md["family"]))
                return isempty(fam) ? "legacy" : fam
            end
        end
    end
    return "legacy"
end

function _model_options(models::Vector{String}=list_models())
    by_family = Dict{String, Vector{String}}()
    for m in models
        f = _spec_family(get_model(m))
        push!(get!(by_family, f, String[]), m)
    end
    options = Any[]
    for family in FAMILY_SORT_ORDER
        models_in_fam = get(by_family, family, String[])
        isempty(models_in_fam) && continue
        label = get(FAMILY_LABELS, family, family)
        push!(options, Dict("label" => "── $(label) ──", "value" => "__hdr__$(family)", "disable" => true))
        for m in sort(models_in_fam)
            push!(options, Dict("label" => m, "value" => m))
        end
    end
    return options
end

function _default_models()
    available = list_models()
    preferred = ["logistic_growth", "gompertz_growth"]
    chosen = [m for m in preferred if m in available]
    isempty(chosen) && return isempty(available) ? String[] : available[1:min(end, 2)]
    return chosen
end

# ── Core helpers (shared logic from Dash app) ─────────────────────────────────

function _model_latex(model_name::AbstractString)
    m = String(model_name)
    haskey(LATEX_CACHE, m) && return LATEX_CACHE[m]
    spec = get_model(m)
    ptxt = join(string.(spec.param_names), ", ")
    fam = lowercase(_spec_family(spec))
    if fam == "logistic"
        return "dN/dt = rN(1-N/K)  |  params: $(ptxt)"
    elseif fam == "gompertz"
        return "dN/dt = rN ln(K/N)  |  params: $(ptxt)"
    elseif fam == "theta_logistic"
        return "dN/dt = rN(1-(N/K)^theta) - drug(E)*N  |  params: $(ptxt)"
    else
        return "du/dt = f(u,t;theta,E),  y=h(u)  |  params: $(ptxt)"
    end
end

function _parse_overlay_filename_metadata(path::AbstractString)
    base = basename(String(path))
    stem = replace(base, r"\.[^.]+$" => "")
    parts = split(stem, "___")
    meta = Dict{Symbol,String}()
    for part in parts
        occursin("=", part) || continue
        key, value = split(part, "="; limit=2)
        key_s = Symbol(strip(key))
        value_s = strip(value)
        isempty(value_s) && continue
        meta[key_s] = value_s
    end
    return meta
end

function _fill_missing_metadata_from_filename!(df::DataFrame, path::AbstractString, raw_cols::Set{Symbol})
    meta = _parse_overlay_filename_metadata(path)
    isempty(meta) && return df

    if (:dose in keys(meta)) && !(:dose in raw_cols)
        parsed = tryparse(Float64, meta[:dose])
        if !isnothing(parsed)
            df[!, :dose] = fill(parsed, nrow(df))
        end
    end

    if (:treatment_amount in keys(meta)) && !(:treatment_amount in raw_cols)
        parsed = tryparse(Float64, meta[:treatment_amount])
        if !isnothing(parsed)
            df[!, :treatment_amount] = fill(parsed, nrow(df))
        end
    end

    if (:cell_line in keys(meta)) && !(:cell_line in raw_cols)
        df[!, :cell_line] = fill(meta[:cell_line], nrow(df))
    end

    if (:density in keys(meta)) && !(:density in raw_cols)
        parsed = tryparse(Float64, meta[:density])
        if !isnothing(parsed)
            df[!, :density] = fill(parsed, nrow(df))
        end
    end

    if (:replicate in keys(meta)) && !(:replicate in raw_cols)
        parsed = tryparse(Int, meta[:replicate])
        if !isnothing(parsed)
            df[!, :replicate] = fill(parsed, nrow(df))
        end
    end

    return df
end

function _safe_load(path::AbstractString)
    p = String(path)
    sig = let s = stat(p)
        string(s.mtime) * ":" * string(s.size)
    end

    cached = get(SAFE_LOAD_CACHE, p, nothing)
    if !isnothing(cached) && cached[1] == sig
        return copy(cached[2])
    end

    raw = CSV.read(p, DataFrame)
    raw_cols = Set(Symbol.(names(raw)))
    normalized = normalize_schema(raw)
    normalized = _fill_missing_metadata_from_filename!(normalized, p, raw_cols)
    SAFE_LOAD_CACHE[p] = (sig, copy(normalized))
    return normalized
end

function _has_stage_metadata(df::DataFrame)
    cols = Set(Symbol.(names(df)))
    return (:culture_type in cols) && (:population_type in cols)
end

function _condition_cols_for_plot(df::DataFrame)
    cols = Set(Symbol.(names(df)))
    preferred = [:dose, :cell_line, :density, :replicate]
    return [c for c in preferred if c in cols]
end

function _plot_df_with_condition(df::DataFrame)
    cols = _condition_cols_for_plot(df)
    out = copy(df)
    if isempty(cols)
        out[!, :condition_label] = fill("all_data", nrow(out))
        return out
    end
    out[!, :condition_label] = [join(["$(c)=$(out[i, c])" for c in cols], " | ") for i in 1:nrow(out)]
    return out
end

function _condition_key(name::AbstractString)
    lowercase(strip(String(name)))
end

function _find_condition(conditions::Vector{FitCondition}, name::AbstractString)
    key = _condition_key(name)
    findfirst(c -> _condition_key(c.name) == key, conditions)
end

function _split_indices(n::Int, train_fraction::Float64; mode::Symbol = :temporal, seed::Int = 42)
    n_train = clamp(round(Int, train_fraction * n), 2, n - 2)
    if mode == :random
        rng = MersenneTwister(seed)
        idx = shuffle(rng, collect(1:n))
        train_idx = sort(idx[1:n_train])
        val_idx   = sort(idx[(n_train + 1):end])
    else
        train_idx = collect(1:n_train)
        val_idx   = collect((n_train + 1):n)
    end
    return train_idx, val_idx
end

function _slice_condition(cond::FitCondition, idx::Vector{Int})
    return FitCondition(cond.name, cond.time[idx], cond.count[idx], cond.error[idx],
        [max(1e-12, cond.count[idx[1]])], cond.exposure, cond.metadata)
end

function _rmse(y::Vector{Float64}, yhat::Vector{Float64})
    return sqrt(mean((y .- yhat) .^ 2))
end

function _mape(y::Vector{Float64}, yhat::Vector{Float64})
    return 100.0 * mean(abs.((y .- yhat) ./ max.(abs.(y), 1e-8)))
end

function _fit_dose_for_condition(cond::FitCondition)
    return evaluate_exposure(cond.exposure, cond.time[1])
end

function _bootstrap_interval(spec::ModelSpec, cond_train::FitCondition, cond_full::FitCondition;
                              n_boot::Int = 30, maxiters::Int = 150)
    n = length(cond_train.time)
    n < 4 && return nothing
    rng   = MersenneTwister(42)
    preds = Vector{Vector{Float64}}()
    dose  = _fit_dose_for_condition(cond_train)
    for _ in 1:n_boot
        idx   = rand(rng, 1:n, n)
        xb    = cond_train.time[idx]; yb = cond_train.count[idx]
        order = sortperm(xb); xb = xb[order]; yb = yb[order]
        try
            fr  = fit_model(spec, xb, yb, dose; maxiters=maxiters)
            sim = simulate(spec, cond_full.time, fr.params; u0=cond_full.u0, exposure=cond_full.exposure)
            if sim.success && length(sim.observed) == length(cond_full.time) && all(isfinite, sim.observed)
                push!(preds, Float64.(sim.observed))
            end
        catch; end
    end
    length(preds) < 5 && return nothing
    mat   = hcat(preds...)
    lower = [quantile(vec(mat[i, :]), 0.025) for i in 1:size(mat, 1)]
    upper = [quantile(vec(mat[i, :]), 0.975) for i in 1:size(mat, 1)]
    return (lower=lower, upper=upper, n_success=length(preds))
end

# ── Parse / validate helpers ──────────────────────────────────────────────────

function _parse_symbol_csv(text, field_name::AbstractString)
    isnothing(text) && throw(ArgumentError("$(field_name) is required"))
    items = [strip(part) for part in split(String(text), ",") if !isempty(strip(part))]
    isempty(items) && throw(ArgumentError("$(field_name) cannot be empty"))
    return Symbol.(items)
end

function _parse_float_csv(text, field_name::AbstractString)
    isnothing(text) && throw(ArgumentError("$(field_name) is required"))
    items = [strip(part) for part in split(String(text), ",") if !isempty(strip(part))]
    isempty(items) && throw(ArgumentError("$(field_name) cannot be empty"))
    return [parse(Float64, item) for item in items]
end

function _parse_state_names(text)
    return _parse_symbol_csv(text, "State names")
end

function _parse_constants_csv(text)
    (isnothing(text) || isempty(strip(String(text)))) && return Symbol[], Float64[]
    names_out = Symbol[]; vals_out = Float64[]
    for item in [strip(part) for part in split(String(text), ",") if !isempty(strip(part))]
        occursin("=", item) || throw(ArgumentError("Constants need `name=value` format, got: $(item)"))
        lhs, rhs = split(item, "="; limit=2)
        push!(names_out, Symbol(strip(lhs))); push!(vals_out, parse(Float64, strip(rhs)))
    end
    return names_out, vals_out
end

function _validate_rhs_expr(expr, allowed::Set{Symbol})
    expr isa Number && return nothing
    if expr isa Symbol
        expr in allowed || throw(ArgumentError("Unsupported symbol: $(expr)"))
        return nothing
    end
    if expr isa Expr && expr.head == :call
        fn = expr.args[1]
        ok = Set([:+, :-, :*, :/, :^, :log, :exp, :sqrt, :sin, :cos, :tan, :abs, :max, :min])
        (fn isa Symbol && fn in ok) || throw(ArgumentError("Unsupported function/op: $(fn)"))
        for a in expr.args[2:end]; _validate_rhs_expr(a, allowed); end
        return nothing
    end
    throw(ArgumentError("Unsupported expression: $(expr)"))
end

function _parse_equation_lines(text, state_names::Vector{Symbol})
    isnothing(text) && throw(ArgumentError("State equations are required"))
    lines = [strip(line) for line in split(String(text), '\n') if !isempty(strip(line))]
    isempty(lines) && throw(ArgumentError("State equations cannot be empty"))
    eq_map = Dict{Symbol,String}()
    for line in lines
        occursin("=", line) || throw(ArgumentError("Equation must have form `state = expression`"))
        lhs, rhs = split(line, "="; limit=2)
        s = Symbol(strip(lhs))
        s in state_names || throw(ArgumentError("Equation for unknown state: $(s)"))
        eq_map[s] = strip(rhs)
    end
    for s in state_names
        haskey(eq_map, s) || throw(ArgumentError("Missing equation for state $(s)"))
    end
    return eq_map
end

function _compile_expression(expr_text::AbstractString, state_names::Vector{Symbol},
                              param_names::Vector{Symbol}, constant_names::Vector{Symbol}=Symbol[])
    parsed = try Meta.parse(String(expr_text))
             catch err; throw(ArgumentError("Cannot parse `$(expr_text)`: $(sprint(showerror, err))")) end
    allowed = Set(vcat(state_names, [:t, :E, :pi], param_names, constant_names))
    _validate_rhs_expr(parsed, allowed)
    fn_expr = Expr(:->, Expr(:tuple, state_names..., :t, :E, param_names..., constant_names...), parsed)
    return Base.eval(@__MODULE__, fn_expr)
end

function _rhs_to_tex(rhs::AbstractString)
    replace(String(rhs), "log(" => "log(", "exp(" => "exp(", "*" => " ")
end

function _builder_template_data(template::AbstractString)
    if template == "sensitive_resistant"
        return (family="mechanistic", states="S, R", observable="S + R",
            params="rS, rR, K, kSR, emax, ic50, hill", constants="",
            lower="1e-6, 1e-6, 1e-3, 0.0, 0.0, 1e-8, 0.1",
            upper="5.0, 5.0, 1e7, 2.0, 20.0, 1e4, 8.0",
            equations="S = rS*S*(1 - (S + R)/K) - kSR*S - emax*(E^hill/(ic50^hill + E^hill))*S\nR = rR*R*(1 - (S + R)/K) + kSR*S")
    elseif template == "lotka_volterra"
        return (family="coculture", states="S, R", observable="S + R",
            params="rS, KS, alphaSR, rR, KR, alphaRS", constants="",
            lower="1e-6, 1e-3, 0.0, 1e-6, 1e-3, 0.0", upper="5.0, 1e7, 5.0, 5.0, 1e7, 5.0",
            equations="S = rS*S*(1 - (S + alphaSR*R)/KS)\nR = rR*R*(1 - (R + alphaRS*S)/KR)")
    elseif template == "theta_hill"
        return (family="theta_logistic", states="N", observable="N",
            params="r, K, theta, ic50, hill", constants="",
            lower="1e-6, 1e-3, 0.1, 1e-8, 0.1", upper="5.0, 1e7, 5.0, 1e4, 8.0",
            equations="N = r*N*(1 - (N/K)^theta) * (1 - E^hill/(ic50^hill + E^hill))")
    elseif template == "gompertz"
        return (family="gompertz", states="N", observable="N",
            params="a, b, K", constants="",
            lower="1e-6, 1e-6, 1e-3", upper="5.0, 10.0, 1e7",
            equations="N = a*N*log(K/N)")
    end
    return (family="logistic", states="N", observable="N",
        params="r, K", constants="",
        lower="1e-6, 1e-3", upper="5.0, 1e7",
        equations="N = r*N*(1 - N/K)")
end

function _build_custom_model_spec(name, family, state_text, observable_text, params_text,
                                   constants_text, lower_text, upper_text, equations_text)
    model_name = strip(String(name))
    isempty(model_name) && throw(ArgumentError("Model name is required"))
    state_names     = _parse_state_names(state_text)
    param_names     = _parse_symbol_csv(params_text, "Parameter names")
    const_names, const_vals = _parse_constants_csv(constants_text)
    lower_bounds    = _parse_float_csv(lower_text, "Lower bounds")
    upper_bounds    = _parse_float_csv(upper_text, "Upper bounds")
    overlap = intersect(Set(param_names), Set(const_names))
    isempty(overlap) || throw(ArgumentError("Cannot be both param and constant: $(join(string.(collect(overlap)), ", "))"))
    length(param_names) == length(lower_bounds) == length(upper_bounds) ||
        throw(ArgumentError("Parameter names, lower bounds, upper bounds must have same length"))
    eq_map   = _parse_equation_lines(equations_text, state_names)
    rhs_fns  = Dict(s => _compile_expression(eq_map[s], state_names, param_names, const_names) for s in state_names)
    obs_expr = isnothing(observable_text) || isempty(strip(String(observable_text))) ?
                   string(first(state_names)) : strip(String(observable_text))
    obs_fn   = _compile_expression(obs_expr, state_names, param_names, const_names)
    ode_fn   = function (du, u, p, t, exposure)
        E  = exposure(t)
        sv = Tuple(u[i] for i in eachindex(state_names))
        for (idx, s) in enumerate(state_names)
            du[idx] = rhs_fns[s](sv..., t, E, p..., const_vals...)
        end
    end
    bounds  = [(lower_bounds[i], upper_bounds[i]) for i in eachindex(param_names)]
    fam     = isnothing(family) || isempty(strip(String(family))) ? "custom" : strip(String(family))
    metadata = Dict(
        :source => :gui_builder,
        :family => fam,
        :equation_rhs => join(["$(s)=$(eq_map[s])" for s in state_names], "; "),
        :observable_expr => obs_expr,
        :preset_constants => Dict(string(const_names[i]) => const_vals[i] for i in eachindex(const_names)),
    )

    # Support both newer and older GrowthParameterEstimation ModelSpec constructors.
    try
        return ModelSpec(
            name=model_name, ode! = ode_fn, param_names=param_names, bounds=bounds,
            n_states=length(state_names),
            observable=u -> obs_fn(Tuple(u[i] for i in eachindex(state_names))..., 0.0, 0.0,
                                   zeros(length(param_names))..., const_vals...),
            base_growth_family=fam, state_names=state_names,
            metadata=metadata)
    catch err
        err isa MethodError || rethrow()
        return ModelSpec(
            String(model_name),
            ode_fn,
            param_names,
            bounds,
            state_names,
            (u, _args...) -> obs_fn(Tuple(u[i] for i in eachindex(state_names))..., 0.0, 0.0,
                                    zeros(length(param_names))..., const_vals...),
            :ode,
            metadata,
        )
    end
end

function _save_gui_model_to_file(name, family, states, observable, params, constants, lower, upper, equations)
    data = isfile(GUI_CUSTOM_MODELS_PATH) ? TOML.parsefile(GUI_CUSTOM_MODELS_PATH) : Dict{String,Any}()
    haskey(data, "models") || (data["models"] = Dict{String,Any}())
    data["models"][name] = Dict{String,Any}(
        "family"=>(String(family)), "states"=>(String(states)), "observable"=>(String(observable)),
        "params"=>(String(params)), "constants"=>(String(constants)),
        "lower"=>(String(lower)), "upper"=>(String(upper)), "equations"=>(String(equations)))
    open(GUI_CUSTOM_MODELS_PATH, "w") do io; TOML.print(io, data); end
end

function _load_gui_models_from_file()
    isfile(GUI_CUSTOM_MODELS_PATH) || return
    data = try TOML.parsefile(GUI_CUSTOM_MODELS_PATH) catch; return end
    models_section = get(data, "models", nothing)
    isnothing(models_section) && return
    n_loaded = 0
    for (mname, mdata) in models_section
        try
            spec = _build_custom_model_spec(
                get(mdata, "name", mname), get(mdata, "family", "custom"),
                get(mdata, "states", "N"), get(mdata, "observable", "N"),
                get(mdata, "params", "r, K"), get(mdata, "constants", ""),
                get(mdata, "lower", "1e-6, 1e-3"), get(mdata, "upper", "5.0, 1e7"),
                get(mdata, "equations", "N = 0.0"))
            register_model!(spec; overwrite=true); n_loaded += 1
        catch err
            @warn "Could not restore GUI model '$(mname)': $(sprint(showerror, err))"
        end
    end
    n_loaded > 0 && @info "Restored $(n_loaded) GUI-built model(s) from $(GUI_CUSTOM_MODELS_PATH)"
end

# ── HTML output helpers ───────────────────────────────────────────────────────

function _wrap_card(content::AbstractString; title::AbstractString="")
    hdr = isempty(title) ? "" :
        "<div style='background:#0f766e;color:#fff;font-weight:bold;padding:8px 16px;font-size:14px'>$(title)</div>"
    return "<div style='background:#f9fafb;border:1px solid #d1fae5;border-radius:6px;overflow:hidden;margin-bottom:20px'>" *
           hdr * "<div style='padding:16px'>$(content)</div></div>"
end

function _df_to_html(df::DataFrame; limit::Int = 15)
    isempty(df) && return "<p style='color:#6b7280'>(empty)</p>"
    shown = nrow(df) > limit ? first(df, limit) : df
    th   = join(["<th style='padding:4px 8px;background:#f0fdf4;font-weight:600;font-size:11px;text-align:left;border:1px solid #e5e7eb'>$(c)</th>" for c in names(shown)], "")
    rows = join(["<tr>$(join(["<td style='padding:3px 8px;font-size:11px;border:1px solid #e5e7eb'>$(shown[i, c])</td>" for c in names(shown)], ""))</tr>" for i in 1:nrow(shown)], "")
    note = nrow(df) > limit ? "<br><small style='color:#6b7280'>Showing $(nrow(shown)) of $(nrow(df)) rows</small>" : ""
    return "<div style='overflow-x:auto'><table style='width:100%;border-collapse:collapse'><thead><tr>$(th)</tr></thead><tbody>$(rows)</tbody></table>$(note)</div>"
end

function _preflight_to_html(path::AbstractString)
    df    = _safe_load(path)
    pf    = preflight_data_quality(df)
    nc    = try
        length(build_conditions(df))
    catch
        0
    end
    stgd  = _has_stage_metadata(df)
    title = "Preflight — $(basename(path)) ($(nc) condition$(nc==1 ? "" : "s")$(stgd ? ", staged" : ""))"
    issues_html = isempty(pf.issues) ? "<p style='color:#0f766e'>&#10003; No data quality issues found.</p>" :
        "<ul>" * join([
            "<li style='color:$(r.severity == "error" ? "#dc2626" : "#d97706");margin-bottom:4px'>$(r.severity): $(r.message)</li>"
            for r in eachrow(pf.issues)
        ], "") * "</ul>"
    return _wrap_card(
        "<h6 style='color:#0f766e;margin:0 0 8px 0'>Data Quality Summary</h6>" *
        _df_to_html(pf.summary; limit=20) *
        "<h6 style='color:#0f766e;margin:12px 0 8px 0'>Condition Quality</h6>" *
        _df_to_html(pf.condition_quality; limit=30) *
        "<h6 style='color:#0f766e;margin:12px 0 8px 0'>Issues</h6>" * issues_html;
        title=title)
end

function _overview_plotdata(df::DataFrame)
    plot_df = _plot_df_with_condition(df)
    traces  = PlotData[]
    grouped = groupby(plot_df, :condition_label)
    for g in grouped
        gs = sort(g, :time)
        xvals = Float64[]
        yvals = Float64[]
        for i in 1:nrow(gs)
            t = gs.time[i]
            y = gs.count[i]
            if ismissing(t) || ismissing(y)
                continue
            end
            tf = try
                Float64(t)
            catch
                continue
            end
            yf = try
                Float64(y)
            catch
                continue
            end
            if isfinite(tf) && isfinite(yf)
                push!(xvals, tf)
                push!(yvals, yf)
            end
        end
        isempty(xvals) && continue
        push!(traces, PlotData(
            x    = xvals,
            y    = yvals,
            mode = "lines+markers",
            name = String(gs.condition_label[1])))
    end
    layout = PlotLayout(title=PlotLayoutTitle(text="Raw trajectories by condition"))
    return traces, layout
end

function _append_unique_path(paths::Vector{String}, path::AbstractString)
    p = String(path)
    isempty(p) && return paths
    p in paths || push!(paths, p)
    return paths
end

function _upload_entries(fileuploads)
    entries = NamedTuple[]
    seen = Set{String}()
    isnothing(fileuploads) && return entries

    path_keys = ("path", "tmp_path", "tempPath", "tmpname", "filepath", "file")
    name_keys = ("name", "filename", "originalname", "fileName")
    child_keys = ("files", "items", "uploads", "value", "data")

    function _dict_get(d::Dict{String,Any}, keys)
        for k in keys
            if haskey(d, k)
                return d[k]
            end
        end
        return nothing
    end

    function _normalize_item(item)
        if item isa Dict
            out = Dict{String,Any}()
            for (k, v) in pairs(item)
                out[string(k)] = v
            end
            return out
        end
        return try
            raw = Dict(item)
            out = Dict{String,Any}()
            for (k, v) in pairs(raw)
                out[string(k)] = v
            end
            out
        catch
            nothing
        end
    end

    function _collect(item)
        if item isa AbstractVector
            for child in item
                _collect(child)
            end
            return
        end

        data = _normalize_item(item)
        isnothing(data) && return

        path_any = _dict_get(data, path_keys)
        name_any = _dict_get(data, name_keys)

        if !isnothing(path_any)
            path = strip(String(path_any))
            if !isempty(path) && !(path in seen)
                name = isnothing(name_any) ? basename(path) : strip(String(name_any))
                isempty(name) && (name = basename(path))
                push!(entries, (path=path, name=name))
                push!(seen, path)
            end
        end

        for ck in child_keys
            if haskey(data, ck)
                _collect(data[ck])
            end
        end
    end

    _collect(fileuploads)
    return entries
end

function _materialize_uploaded_path(path::AbstractString, name::AbstractString)
    src = String(path)
    isfile(src) || return src

    original_name = strip(String(name))
    isempty(original_name) && return src

    safe_name = replace(original_name, r"[\\/:*?\"<>|]" => "_")
    upload_dir = joinpath(tempdir(), "gpe_gui_uploads")
    mkpath(upload_dir)
    dst = joinpath(upload_dir, safe_name)

    if !isfile(dst)
        cp(src, dst; force=true)
        return dst
    end

    try
        if filesize(dst) == filesize(src)
            return dst
        end
    catch
    end

    stem, ext = splitext(safe_name)
    alt = joinpath(upload_dir, "$(stem)_$(rand(UInt32))$(ext)")
    cp(src, alt; force=true)
    return alt
end

function _uploaded_csv_options(paths::Vector{String})
    isempty(paths) && return Any[Dict("label" => "No uploaded files yet", "value" => "")]
    return Any[Dict("label" => basename(p), "value" => p) for p in paths]
end

function _uploaded_files_html(paths::Vector{String}; active_path::AbstractString = "")
    isempty(paths) && return "<p style='color:#6b7280'>No files uploaded yet. Add one or more CSV files above.</p>"

    rows = NamedTuple[]
    for p in paths
        if !isfile(p)
            push!(rows, (active=(p == active_path ? "yes" : ""), file=basename(p), path=p, rows=0, conditions=0, staged="unknown", status="missing"))
            continue
        end
        try
            df = _safe_load(p)
            ncond = try
                length(build_conditions(df))
            catch
                0
            end
            staged = _has_stage_metadata(df) ? "yes" : "no"
            push!(rows, (active=(p == active_path ? "yes" : ""), file=basename(p), path=p, rows=nrow(df), conditions=ncond, staged=staged, status="ok"))
        catch err
            push!(rows, (active=(p == active_path ? "yes" : ""), file=basename(p), path=p, rows=0, conditions=0, staged="unknown", status="error: $(sprint(showerror, err))"))
        end
    end

    return _df_to_html(DataFrame(rows); limit=100)
end

function _float_col(df::DataFrame, col::Symbol)
    vals = Float64[]
    for v in df[!, col]
        ismissing(v) && continue
        f = try Float64(v) catch; continue end
        isfinite(f) && push!(vals, f)
    end
    return vals
end

function _overview_plotdata_multi(paths::Vector{String})
    traces = PlotData[]
    loaded = 0
    for p in paths
        isfile(p) || continue
        df = try _safe_load(p) catch; continue end
        loaded += 1
        fname = basename(p)

        plot_df = _plot_df_with_condition(df)
        grouped = groupby(plot_df, :condition_label)
        for g in grouped
            gs = sort(g, :time)
            xvals = Float64[]
            yvals = Float64[]
            for i in 1:nrow(gs)
                t = gs.time[i]; y = gs.count[i]
                ismissing(t) || ismissing(y) && continue
                tf = try Float64(t) catch; continue end
                yf = try Float64(y) catch; continue end
                if isfinite(tf) && isfinite(yf)
                    push!(xvals, tf); push!(yvals, yf)
                end
            end
            cond_label = String(gs.condition_label[1])
            if !isempty(xvals)
                push!(traces, PlotData(
                    x = xvals,
                    y = yvals,
                    mode = "lines+markers",
                    name = "$(fname) :: $(cond_label)",
                ))
            end
        end
    end

    layout = PlotLayout(title=PlotLayoutTitle(text="Raw trajectories across uploaded files ($(loaded))"))
    return traces, layout
end

function _premade_csv_path(choice::AbstractString)
    key = String(choice)
    if key == "staged_monoculture"
        return joinpath(EXAMPLE_DIR, "staged_monoculture.csv")
    elseif key == "coculture_stages"
        return joinpath(EXAMPLE_DIR, "coculture_stages.csv")
    end
    return joinpath(EXAMPLE_DIR, "basic_pipeline.csv")
end

function _selected_data_source_path(source_mode::AbstractString, active_uploaded_csv::AbstractString, premade_csv_choice::AbstractString, csv_path_input::AbstractString)
    mode = String(source_mode)
    if mode == "premade"
        return _premade_csv_path(premade_csv_choice)
    elseif mode == "filepath"
        return String(csv_path_input)
    end
    return String(active_uploaded_csv)
end

function _combine_uploaded_data(paths::Vector{String})
    frames = DataFrame[]
    for p in paths
        isfile(p) || continue
        df = _safe_load(p)
        df[!, :source_file] = fill(basename(p), nrow(df))
        push!(frames, df)
    end
    isempty(frames) && return DataFrame()
    return reduce((a, b) -> vcat(a, b; cols=:union), frames)
end

function _multi_preflight_to_html(paths::Vector{String})
    summary_rows = NamedTuple[]
    issue_rows = NamedTuple[]
    for p in paths
        isfile(p) || continue
        fname = basename(p)
        try
            df = _safe_load(p)
            pf = preflight_data_quality(df)
            ncond = try
                length(build_conditions(df))
            catch
                0
            end
            nerr = nrow(filter(:severity => ==("error"), pf.issues))
            nwarn = nrow(filter(:severity => ==("warning"), pf.issues))
            push!(summary_rows, (
                file=fname,
                rows=nrow(df),
                conditions=ncond,
                staged=_has_stage_metadata(df) ? "yes" : "no",
                issues=nrow(pf.issues),
                errors=nerr,
                warnings=nwarn,
            ))
            for r in eachrow(first(pf.issues, min(12, nrow(pf.issues))))
                push!(issue_rows, (file=fname, severity=String(r.severity), message=String(r.message)))
            end
        catch err
            push!(summary_rows, (
                file=fname,
                rows=0,
                conditions=0,
                staged="unknown",
                issues=1,
                errors=1,
                warnings=0,
            ))
            push!(issue_rows, (file=fname, severity="error", message=sprint(showerror, err)))
        end
    end

    summary_df = isempty(summary_rows) ? DataFrame(file=String[], rows=Int[], conditions=Int[], staged=String[], issues=Int[], errors=Int[], warnings=Int[]) : DataFrame(summary_rows)
    issues_df = isempty(issue_rows) ? DataFrame(file=String[], severity=String[], message=String[]) : DataFrame(issue_rows)

    return _wrap_card(
        "<h6 style='color:#0f766e;margin:0 0 8px 0'>Per-file Preflight Summary</h6>" *
        _df_to_html(summary_df; limit=100) *
        "<h6 style='color:#0f766e;margin:12px 0 8px 0'>Top Issues Across Uploaded Files</h6>" *
        _df_to_html(issues_df; limit=120);
        title="Preflight - Upload Collection")
end

function _rank_output_html(path, models, cond_name, n_starts, maxiters, train_fraction, split_mode, n_boot)
    df         = _safe_load(path)
    conditions = build_conditions(df)
    isempty(conditions) && return (
        _wrap_card("<p style='color:#d97706'>No conditions built.</p>"; title="Ranking Results"),
        PlotData[], PlotLayout(title=PlotLayoutTitle(text="No conditions")))
    isempty(models) && (models = _default_models())

    split_frac = clamp(Float64(train_fraction), 0.5, 0.95)
    split_sym  = split_mode == "random" ? :random : :temporal

    split_rows = NamedTuple[]; train_conds = FitCondition[]; val_conds = Dict{String,FitCondition}()
    for cond in conditions
        n_pts = length(cond.time)
        if n_pts < 4
            push!(train_conds, cond)
            push!(split_rows, (condition=cond.name, n_total=n_pts, n_train=n_pts, n_val=0, split_applied=false))
            continue
        end
        ti, vi = _split_indices(n_pts, split_frac; mode=split_sym, seed=42)
        push!(train_conds, _slice_condition(cond, ti))
        val_conds[cond.name] = _slice_condition(cond, vi)
        push!(split_rows, (condition=cond.name, n_total=n_pts, n_train=length(ti), n_val=length(vi), split_applied=true))
    end
    split_df = DataFrame(split_rows)

    ranked = rank_models(models, train_conds; n_starts=n_starts, maxiters=maxiters,
                         top_k=min(length(models), 5), seed=42)

    val_rows = NamedTuple[]
    for model_name in ranked.ranking.model
        if !haskey(ranked.fits, model_name)
            push!(val_rows, (model=model_name, val_rmse=Inf, val_mape=Inf, val_points=0)); continue
        end
        spec = get_model(model_name); fi = ranked.fits[model_name]
        val_obs = Float64[]; val_pred = Float64[]
        for (cname, cv) in pairs(val_conds)
            hit = findfirst(pc -> _condition_key(pc.condition) == _condition_key(cname) && pc.success, fi.per_condition)
            isnothing(hit) && continue
            p   = fi.per_condition[hit].params
            sim = simulate(spec, cv.time, p; u0=cv.u0, exposure=cv.exposure)
            if sim.success && length(sim.observed) == length(cv.count)
                append!(val_obs, cv.count); append!(val_pred, Float64.(sim.observed))
            end
        end
        if isempty(val_obs)
            push!(val_rows, (model=model_name, val_rmse=Inf, val_mape=Inf, val_points=0))
        else
            push!(val_rows, (model=model_name, val_rmse=_rmse(val_obs, val_pred),
                val_mape=_mape(val_obs, val_pred), val_points=length(val_obs)))
        end
    end
    val_df = DataFrame(val_rows); sort!(val_df, :val_rmse)
    ranking_table = leftjoin(ranked.ranking, val_df; on=:model)
    sort!(ranking_table, [:val_rmse, :bic])

    # Fit figure
    sel_cond = isempty(cond_name) ? conditions[1].name : cond_name
    cidx = _find_condition(conditions, sel_cond)
    isnothing(cidx) && (cidx = 1; sel_cond = conditions[1].name)
    cond = conditions[cidx]

    traces = PlotData[PlotData(x=Float64.(cond.time), y=Float64.(cond.count),
        mode="markers", name="Observed",
        marker=PlotDataMarker(size=9, color="#111827"))]

    for model_name in ranking_table.model
        haskey(ranked.fits, model_name) || continue
        fi  = ranked.fits[model_name]
        hit = findfirst(pc -> _condition_key(pc.condition) == _condition_key(sel_cond) && pc.success, fi.per_condition)
        isnothing(hit) && continue
        p   = fi.per_condition[hit].params
        sim = simulate(get_model(model_name), cond.time, p; u0=cond.u0, exposure=cond.exposure)
        sim.success || continue
        push!(traces, PlotData(x=Float64.(cond.time), y=Float64.(sim.observed),
            mode="lines", name=model_name, line=PlotlyLine(width=2)))
    end

    # Bootstrap CI
    interval_success = 0
    if !isempty(ranking_table)
        bm = String(ranking_table.model[1])
        if haskey(ranked.fits, bm) && haskey(val_conds, sel_cond)
            t_idx = findfirst(c -> _condition_key(c.name) == _condition_key(sel_cond), train_conds)
            if !isnothing(t_idx)
                ci = _bootstrap_interval(get_model(bm), train_conds[t_idx], cond;
                    n_boot=max(Int(n_boot), 5), maxiters=min(maxiters, 250))
                if !isnothing(ci)
                    interval_success = ci.n_success
                    push!(traces, PlotData(
                        x         = vcat(Float64.(cond.time), reverse(Float64.(cond.time))),
                        y         = vcat(ci.upper, reverse(ci.lower)),
                        fill      = "toself",
                        fillcolor = "rgba(15,118,110,0.15)",
                        line      = PlotlyLine(color="rgba(15,118,110,0.05)"),
                        name      = "95% interval ($(bm))"))
                end
            end
        end
    end

    fig_layout = PlotLayout(title=PlotLayoutTitle(text="Condition: $(sel_cond)"))

    interval_note = interval_success > 0 ?
        "95% predictive interval shown (bootstrap fits retained: $(interval_success))." :
        "No predictive interval (insufficient successful bootstrap fits)."
    rank_html = _wrap_card(
        "<h6 style='color:#0f766e;margin:0 0 8px 0'>Model Ranking (validation-first)</h6>" *
        "<p style='font-size:12px;color:#6b7280'>Ordered by validation RMSE first, then BIC.</p>" *
        _df_to_html(ranking_table; limit=20) *
        "<h6 style='color:#0f766e;margin:12px 0 8px 0'>Train/Validation Split</h6>" *
        _df_to_html(split_df; limit=30) *
        "<p style='font-size:13px;margin-top:8px'>$(interval_note)</p>";
        title="Ranking Results")

    return rank_html, traces, fig_layout
end

function _pipeline_output_html(path, models)
    df  = _safe_load(path)
    isempty(models) && (models = _default_models())
    cfg = default_config(output_dir="results/gui_pipeline")
    run = run_pipeline(df; config=cfg, include_models=models,
        strict_schema=false, qc_before_fit=true, preflight_before_fit=true)
    nf  = nrow(run.failures)
    return _wrap_card(
        "<p style='background:#f0fdf4;border:1px solid #6ee7b7;padding:10px;border-radius:6px'>" *
        "Pipeline complete. Ranked: $(nrow(run.ranking)) | Failures: $(nf)</p>" *
        "<h6 style='color:#0f766e;margin:12px 0 8px 0'>Full Model Ranking</h6>" *
        _df_to_html(run.ranking; limit=30) *
        (nf > 0 ? "<h6 style='color:#0f766e;margin:12px 0 8px 0'>Failures</h6>" * _df_to_html(run.failures; limit=20) : "");
        title="Full Pipeline Results")
end

function _manual_stage_name(raw_name, idx::Int, used::Set{String})
    base = strip(String(raw_name))
    isempty(base) && (base = "stage_$(idx)")
    candidate = base
    suffix = 2
    while candidate in used
        candidate = "$(base)_$(suffix)"
        suffix += 1
    end
    push!(used, candidate)
    return candidate
end

function _model_param_names(model_name::AbstractString)
    name = String(model_name)
    if isempty(name) || !(name in list_models())
        return String[]
    end
    return String.(get_model(name).param_names)
end

function _infer_prev_stage_mapping(prev_stage::String, prev_model::String, next_model::String)
    inherited = Dict{Symbol,Tuple{String,Symbol}}()
    src_params = _model_param_names(prev_model)
    dst_params = _model_param_names(next_model)
    if isempty(src_params) || isempty(dst_params)
        return inherited
    end

    src_lower = Dict(lowercase(p) => p for p in src_params)
    for dst in dst_params
        dlow = lowercase(dst)
        if haskey(src_lower, dlow)
            inherited[Symbol(dst)] = (prev_stage, Symbol(src_lower[dlow]))
            continue
        end
        partial = findfirst(s -> occursin(dlow, lowercase(s)) || occursin(lowercase(s), dlow), src_params)
        if !isnothing(partial)
            inherited[Symbol(dst)] = (prev_stage, Symbol(src_params[partial]))
        end
    end
    return inherited
end

function _parse_mapping_value(value, default_stage::String)
    text = strip(String(value))
    isempty(text) && return nothing
    if occursin(".", text)
        parts = split(text, "."; limit=2)
        stage_name = strip(parts[1])
        param_name = strip(parts[2])
        if isempty(stage_name) || isempty(param_name)
            return nothing
        end
        return (stage_name, Symbol(param_name))
    end
    isempty(default_stage) && return nothing
    return (default_stage, Symbol(text))
end

function _manual_stage_input(pipeline_stages)
    isempty(pipeline_stages) && error("No manual stages configured")

    stage_col = :__manual_stage
    used_names = Set{String}()
    stage_defs = NamedTuple[]
    stage_dfs = DataFrame[]

    for (i, sraw) in enumerate(pipeline_stages)
        s = Dict{String,Any}(String(k) => v for (k, v) in pairs(Dict(sraw)))
        stage_name = _manual_stage_name(get(s, "name", "Stage $(i)"), i, used_names)
        csv_file = strip(String(get(s, "csv_file", "")))
        model_name = strip(String(get(s, "model_name", "")))

        isempty(csv_file) && error("Stage $(i) ($(stage_name)) is missing a CSV file")
        isfile(csv_file) || error("Stage $(i) ($(stage_name)) CSV not found: $(csv_file)")
        isempty(model_name) && error("Stage $(i) ($(stage_name)) is missing a model name")
        model_name in list_models() || error("Stage $(i) ($(stage_name)) uses unknown model: $(model_name)")

        sdf = _safe_load(csv_file)
        sdf[!, stage_col] = fill(stage_name, nrow(sdf))
        push!(stage_dfs, sdf)

        mapping = haskey(s, "param_mapping") ? Dict{String,Any}(string(k) => v for (k, v) in pairs(Dict(s["param_mapping"]))) : Dict{String,Any}()
        push!(stage_defs, (name=stage_name, model=model_name, mapping=mapping))
    end

    combined = reduce((a, b) -> vcat(a, b; cols=:union), stage_dfs)

    workflow_stages = GrowthParameterEstimation.PipelineStage[]
    for i in eachindex(stage_defs)
        spec = stage_defs[i]
        prev_stage_name = i > 1 ? stage_defs[i - 1].name : ""
        prev_model_name = i > 1 ? stage_defs[i - 1].model : ""

        inherited = i > 1 ? _infer_prev_stage_mapping(prev_stage_name, prev_model_name, spec.model) : Dict{Symbol,Tuple{String,Symbol}}()
        for (dest_param, source_ref) in spec.mapping
            parsed = _parse_mapping_value(source_ref, prev_stage_name)
            isnothing(parsed) && continue
            inherited[Symbol(dest_param)] = parsed
        end

        push!(workflow_stages, GrowthParameterEstimation.PipelineStage(
            spec.name,
            "Manual stage $(i) from pipeline designer",
            row -> row[stage_col] == spec.name,
            [stage_col, :treatment_amount, :dose, :cell_line, :population_type, :density, :replicate],
            [spec.model],
            Symbol[],
            Dict{Symbol,Float64}(),
            inherited,
        ))
    end

    return combined, workflow_stages
end

function _staged_output_html(path; pipeline_stages=nothing)
    use_manual = !isnothing(pipeline_stages) && !isempty(pipeline_stages)
    mode_note = use_manual ? "Manual stage assignment (Pipeline Designer order)" : "Default staged templates (treated/untreated filters)"

    df = nothing
    stages = nothing
    if use_manual
        df, stages = _manual_stage_input(pipeline_stages)
    else
        df = _safe_load(path)
    end

    if !use_manual && !_has_stage_metadata(df)
        return "<div style='background:#fffbeb;border:1px solid #fcd34d;padding:10px;border-radius:6px'>" *
               "&#9888; Dataset lacks stage metadata (culture_type, population_type required). " *
               "Load a staged example dataset from Tab 1 or configure manual stages in Tab 2.</div>"
    end

    cfg = default_config(output_dir="results/gui_staged")
    run = if use_manual
        run_staged_pipeline(df; stages=stages, config=cfg, selection_mode=:best_bic,
            strict_schema=false, qc_before_fit=true, preflight_before_fit=false, export_stage_results=true)
    else
        run_staged_pipeline(df; stages=default_stages(), config=cfg, selection_mode=:best_bic,
            strict_schema=false, qc_before_fit=true, preflight_before_fit=true, export_stage_results=true)
    end

    stage_rows = DataFrame(
        stage=[s.name for s in run.stages],
        status=[s.status == "skipped" ? "not_applicable" : s.status for s in run.stages],
        n_conditions=[s.n_conditions for s in run.stages])

    best_rows = NamedTuple[]; rank_rows = NamedTuple[]
    for s in run.stages
        (s.status != "completed" || isnothing(s.result)) && continue
        for r in eachrow(s.result.ranking)
            push!(rank_rows, (stage=String(s.name), model=String(r.model), bic=Float64(r.bic)))
        end
        for (mname, fi) in pairs(s.result.fits), pc in fi.per_condition
            pc.success || continue
            push!(best_rows, (stage=String(s.name), condition=String(pc.condition), model=String(mname)))
        end
    end

    halt = isnothing(run.halted_stage) ? "none" : String(run.halted_stage)
    summary_html = _wrap_card(
        "<p style='background:#f0fdf4;border:1px solid #6ee7b7;padding:10px;border-radius:6px'>" *
        "Mode: $(mode_note)</p>" *
        "<p style='background:#f0fdf4;border:1px solid #6ee7b7;padding:10px;border-radius:6px;margin-top:8px'>" *
        "Staged run complete. Completed: $(run.completed) | Halted at: $(halt)</p>" *
        "<h6 style='color:#0f766e;margin:12px 0 8px 0'>Stage Status</h6>" * _df_to_html(stage_rows; limit=20) *
        (isempty(best_rows) ? "" : "<h6 style='color:#0f766e;margin:12px 0 8px 0'>Best Model By Condition</h6>" *
            _df_to_html(DataFrame(best_rows); limit=100)) *
        (isempty(rank_rows) ? "" : "<h6 style='color:#0f766e;margin:12px 0 8px 0'>Ranking By Stage</h6>" *
            _df_to_html(DataFrame(rank_rows); limit=100));
        title="Staged Pipeline Summary")

    stage_cards = ""
    for s in run.stages
        sname  = String(s.name)
        status = s.status == "skipped" ? "not_applicable" : String(s.status)
        color  = s.status == "completed" ? "#0f766e" : "#6b7280"
        body   = if s.status == "completed" && !isnothing(s.output_dir)
            fig_dir = joinpath(String(s.output_dir), "figures")
            isdir(fig_dir) ? "<p style='font-size:12px;color:#6b7280'>Completed. Results saved to: $(fig_dir)</p>" :
                "<p style='font-size:12px;color:#6b7280'>Completed. Output: $(s.output_dir)</p>"
        else
            "<p style='font-size:12px;color:#6b7280'>Status: $(status)</p>"
        end
        stage_cards *= "<div style='margin-bottom:20px;border-radius:6px;overflow:hidden;border:1px solid #d1fae5'>" *
            "<div style='background:$(color);color:#fff;font-weight:bold;padding:8px 16px;font-size:14px'>Stage: $(uppercase(sname)) [$(status)]</div>" *
            "<div style='padding:12px'>$(body)</div></div>"
    end

    failures_html = nrow(run.failures) > 0 ?
        _wrap_card("<h6>Failures</h6>" * _df_to_html(run.failures; limit=20); title="Failures") : ""

    return summary_html * stage_cards * failures_html
end

function _pipeline_default_model(selected_models)
    if !isnothing(selected_models) && !isempty(selected_models)
        valid = [String(m) for m in selected_models if !startswith(String(m), "__hdr__")]
        !isempty(valid) && return valid[1]
    end
    available = list_models()
    return isempty(available) ? "" : String(available[1])
end

function _pipeline_stage_index(value, fallback::Int = 1)
    if isnothing(value)
        return fallback
    end
    try
        idx = parse(Int, String(value))
        return max(1, idx)
    catch
        return fallback
    end
end

function _pipeline_stage_options(stages)
    isempty(stages) && return Any[Dict("label" => "Add stages first…", "value" => 1)]
    return [Dict("label" => "$(i). $(String(get(Dict(s), "name", "Stage $(i)")))", "value" => i)
            for (i, s) in enumerate(stages)]
end

function _pipeline_stage_cards_html(stages, current_idx::Int)
    if isempty(stages)
        return "<p style='color:#6b7280'>Click 'Add Stage' to begin configuring your pipeline.</p>"
    end
    cards = String[]
    for (i, sraw) in enumerate(stages)
        s = Dict{String,Any}(String(k) => v for (k, v) in pairs(Dict(sraw)))
        sname = String(get(s, "name", "Stage $(i)"))
        sfile = String(get(s, "csv_file", ""))
        smodel = String(get(s, "model_name", ""))
        mapping = haskey(s, "param_mapping") ? Dict{String,Any}(string(k) => v for (k, v) in pairs(Dict(s["param_mapping"]))) : Dict{String,Any}()
        mapping_text = isempty(mapping) ? "<p style='margin:4px 0 0 0;color:#6b7280;font-size:12px'>No parameter mapping saved yet.</p>" :
            "<p style='margin:4px 0 0 0;font-size:12px;color:#374151'><strong>Parameter mapping:</strong> " *
            join(["$(k) → $(v)" for (k, v) in sort(collect(mapping))], ", ") * "</p>"
        push!(cards, "<div style='margin-bottom:14px;border:1px solid #d1fae5;border-left:4px solid $(i == current_idx ? "#0f766e" : "#6b7280");border-radius:6px;padding:12px;background:#fff'>" *
            "<div style='font-weight:700;color:#0f172a'>$(i). $(sname)$(i == current_idx ? " <span style=\"color:#0f766e\">(selected)</span>" : "")</div>" *
            "<div style='font-size:12px;color:#374151;margin-top:4px'>CSV: $(isempty(sfile) ? "(not set)" : sfile)</div>" *
            "<div style='font-size:12px;color:#374151;margin-top:2px'>Model: $(isempty(smodel) ? "(not set)" : smodel)</div>" *
            mapping_text *
            "<div style='font-size:11px;color:#6b7280;margin-top:6px'>Use the stage controls above to reorder, remove, or copy the current data/model into this stage.</div>" *
            "</div>")
    end
    return join(cards, "")
end

function _pipeline_stage_flowchart_html(stages)
    if isempty(stages)
        return "<p style='color:#6b7280;text-align:center;padding:40px'>Stage 1 → Stage 2 → Stage 3 (configured order shown here)</p>"
    end
    flow_text = join([String(get(Dict(s), "name", "Stage $(i)")) for (i, s) in enumerate(stages)], " → ")
    return "<p style='color:#374151;font-weight:600;text-align:center;padding:24px 12px'>$(flow_text)</p>"
end

function _pipeline_mapping_html(stages, current_idx::Int, selected_models)
    if length(stages) < 2
        return "<p style='color:#6b7280'>Add at least two stages to see carry-over mapping suggestions.</p>"
    end
    idx = clamp(current_idx, 1, length(stages))
    if idx >= length(stages)
        return "<p style='color:#6b7280'>The selected stage is the final stage. No next-stage mapping is needed.</p>"
    end
    current = Dict{String,Any}(String(k) => v for (k, v) in pairs(Dict(stages[idx])))
    next_stage = Dict{String,Any}(String(k) => v for (k, v) in pairs(Dict(stages[idx + 1])))
    source_model = isempty(String(get(current, "model_name", ""))) ? _pipeline_default_model(selected_models) : String(get(current, "model_name", ""))
    target_model = String(get(next_stage, "model_name", ""))
    source_params = isempty(source_model) || !(source_model in list_models()) ? String[] : String.(get_model(source_model).param_names)
    target_params = isempty(target_model) || !(target_model in list_models()) ? String[] : String.(get_model(target_model).param_names)
    rows = String[]
    if isempty(source_params) || isempty(target_params)
        push!(rows, "<tr><td style='padding:6px 8px;font-size:12px'>No parameter details available.</td><td></td><td></td></tr>")
    else
        for s in source_params
            exact = findfirst(t -> lowercase(t) == lowercase(s), target_params)
            partial = isnothing(exact) ? findfirst(t -> occursin(lowercase(s), lowercase(t)) || occursin(lowercase(t), lowercase(s)), target_params) : nothing
            mapped = isnothing(exact) ? (isnothing(partial) ? "(unmapped)" : target_params[partial]) : target_params[exact]
            push!(rows, "<tr><td style='padding:6px 8px;font-size:12px'>$(s)</td><td style='padding:6px 8px;font-size:12px;text-align:center'>→</td><td style='padding:6px 8px;font-size:12px'>$(mapped)</td></tr>")
        end
    end
    return "<p style='margin:4px 0 10px 0;color:#374151'>Map fitted variables from <strong>$(isempty(source_model) ? "(current model)" : source_model)</strong> into next stage <strong>$(String(get(next_stage, "name", "Stage $(idx + 1)")))</strong> model <strong>$(isempty(target_model) ? "(not set)" : target_model)</strong>.</p>" *
        "<table style='width:100%;border-collapse:collapse;border:1px solid #e5e7eb'><thead><tr><th style='text-align:left;padding:6px 8px;background:#f0fdf4'>From current stage</th><th style='padding:6px 8px;background:#f0fdf4'></th><th style='text-align:left;padding:6px 8px;background:#f0fdf4'>To next stage</th></tr></thead><tbody>$(join(rows, ""))</tbody></table>" *
        "<small style='color:#6b7280;display:block;margin-top:8px'>This table mirrors the carry-over prompt from the Dash app. Use it to decide parameter transfer before advancing the pipeline.</small>"
end

function _pipeline_refresh_outputs(stages, current_idx::Int, selected_models)
    opts = _pipeline_stage_options(stages)
    idx = isempty(stages) ? 1 : clamp(current_idx, 1, length(stages))
    cards = _pipeline_stage_cards_html(stages, idx)
    flow = _pipeline_stage_flowchart_html(stages)
    mapping = _pipeline_mapping_html(stages, idx, selected_models)
    return opts, idx, cards, flow, mapping
end

function _save_pipeline(pipeline::Pipeline)
    config = if isfile(GUI_PIPELINES_PATH)
        try
            TOML.parsefile(GUI_PIPELINES_PATH)
        catch
            Dict{String,Any}()
        end
    else
        Dict{String,Any}()
    end
    pipelines = haskey(config, "pipelines") ? Dict(config["pipelines"]) : Dict{String,Any}()
    stages_dict = Any[]
    for stage in pipeline.stages
        push!(stages_dict, Dict(
            "name" => stage.name,
            "csv_file" => stage.csv_file,
            "model_name" => stage.model_name,
            "param_mapping" => stage.param_mapping,
        ))
    end
    pipelines[pipeline.name] = Dict("stages" => stages_dict)
    config["pipelines"] = pipelines
    TOML.open(GUI_PIPELINES_PATH, "w") do io
        TOML.print(io, config)
    end
    return nothing
end

# ── Restore saved models on startup ──────────────────────────────────────────
_load_gui_models_from_file()

# ══════════════════════════════════════════════════════════════════════════════
# Reactive model
# ══════════════════════════════════════════════════════════════════════════════

@app GrowthGUI begin

    # === Navigation ===
    @in  active_tab = "tab-load"

    # === Tab 1: Load Data ===
    @in  csv_path_input      = ""
    @in  fileuploads        = Any[]
    @in  active_uploaded_csv = ""
    @in  data_source_mode   = "upload"
    @in  premade_csv_choice = "basic_pipeline"
    @in  btn_visualize_data = 0

    @out csv_path_loaded         = ""
    @out uploaded_csv_paths      = String[]
    @out uploaded_csv_options    = Any[Dict("label" => "No uploaded files yet", "value" => "")]
    @out uploaded_files_html     = "<p style='color:#6b7280'>No files uploaded yet. Add one or more CSV files above.</p>"
    @out status_msg              = "<span style='color:#6b7280'>Select a source, then click Visualize Data.</span>"
    @out data_preview_html       = "<p style='color:#6b7280'>Click Visualize Data to preview the selected file.</p>"
    @out preflight_html          = "<p style='color:#6b7280'>Click Visualize Data to run preflight checks on the selected file.</p>"
    @out conditions_list         = String[]
    @out overview_data::Vector{PlotData}  = PlotData[]
    @out overview_layout::PlotLayout      = PlotLayout(title=PlotLayoutTitle(text="Load data to see trajectories"))
    @out has_stage_metadata_flag = false

    # === Tab 2: Pipeline Designer ===
    @in  pipeline_name         = "my_pipeline"
    @in  pipeline_stage_select = 1
    @in  stage_csv_select      = ""
    @in  btn_add_stage         = 0
    @in  btn_save_pipeline     = 0
    @in  btn_stage_up          = 0
    @in  btn_stage_down        = 0
    @in  btn_stage_remove      = 0
    @in  btn_stage_use_current = 0
    @in  btn_stage_apply_csv   = 0
    @in  btn_stage_next        = 0

    @out pipeline_stages        = Any[]
    @out pipeline_stage_options  = Any[Dict("label" => "Add stages first…", "value" => 1)]
    @out pipeline_stages_html    = "<p style='color:#6b7280'>Click 'Add Stage' to begin configuring your pipeline.</p>"
    @out pipeline_flowchart_html = "<p style='color:#6b7280;text-align:center;padding:40px'>Stage 1 → Stage 2 → Stage 3 (configured order shown here)</p>"
    @out pipeline_mapping_html   = "<p style='color:#6b7280'>Add at least two stages to see carry-over mapping suggestions.</p>"
    @out pipeline_status_html    = "<p style='color:#6b7280'>Stage list, save controls, and carry-over prompts appear here.</p>"

    # === Tab 3: Build Models ===
    @in  builder_template    = "logistic"
    @in  btn_load_template   = 0
    @in  builder_model_name  = "custom_growth_model"
    @in  builder_family      = "custom"
    @in  builder_state_names = "N"
    @in  builder_observable  = "N"
    @in  builder_param_names = "r, K"
    @in  builder_constants   = ""
    @in  builder_lower_bounds = "1e-6, 1e-3"
    @in  builder_upper_bounds = "5.0, 1e7"
    @in  builder_equations    = "N = r*N*(1 - N/K)"
    @in  btn_register_model   = 0

    @out builder_preview_html = "<p style='color:#6b7280'>Enter equations above to see a preview.</p>"
    @out custom_model_status  = "<small style='color:#6b7280'>No model registered yet.</small>"

    # === Tab 4: Select Models ===
    @in  selected_models      = _default_models()
    @in  n_starts             = 8
    @in  maxiters             = 300

    @out model_options_list   = _model_options()
    @out model_equations_html = "<p style='color:#6b7280'>Select models above to see their equations.</p>"

    # === Tab 5: Fit & Rank ===
    @in  condition_select     = ""
    @in  train_frac           = 0.7
    @in  split_mode_v         = "temporal"
    @in  uncertainty_boot     = 30
    @in  btn_run_fit          = 0
    @in  btn_run_pipeline     = 0

    @out rank_html            = "<p style='color:#6b7280'>Click 'Run Fit &amp; Rank' to see results.</p>"
    @out fit_data::Vector{PlotData}  = PlotData[]
    @out fit_layout::PlotLayout      = PlotLayout(title=PlotLayoutTitle(text="Run ranking to see fits"))
    @out pipeline_output_html = "<p style='color:#6b7280'>Run a pipeline to see results here.</p>"

    # === Tab 6: Staged Analysis ===
    @in  btn_run_staged       = 0
    @out staged_html          = "<p style='color:#6b7280'>Run staged analysis to see results here.</p>"

    # ══════════════════════════════════════════════════════════════════════════
    # Watchers
    # ══════════════════════════════════════════════════════════════════════════

    @onchange fileuploads begin
        if !isempty(fileuploads)
            entries = Base.invokelatest(_upload_entries, fileuploads)
            added_paths = String[]
            for entry in entries
                if isfile(entry.path)
                    display_path = Base.invokelatest(_materialize_uploaded_path, entry.path, entry.name)
                    uploaded_csv_paths = Base.invokelatest(_append_unique_path, copy(uploaded_csv_paths), display_path)
                    push!(added_paths, display_path)
                end
            end

            if !isempty(added_paths)
                uploaded_csv_options = Base.invokelatest(_uploaded_csv_options, uploaded_csv_paths)
                active_uploaded_csv = last(added_paths)
                stage_csv_select = isempty(stage_csv_select) ? active_uploaded_csv : (stage_csv_select in uploaded_csv_paths ? stage_csv_select : active_uploaded_csv)
                uploaded_files_html = Base.invokelatest(_uploaded_files_html, uploaded_csv_paths; active_path=active_uploaded_csv)
                traces, layout = Base.invokelatest(_overview_plotdata_multi, uploaded_csv_paths)
                overview_data   = traces
                overview_layout = layout
                status_msg = "<span style='color:#0f766e'>&#10003; Uploaded $(length(added_paths)) file(s). Select source and click Visualize Data when ready.</span>"
            else
                status_msg = "<span style='color:#dc2626'>&#10060; Upload failed: no readable CSV paths received.</span>"
            end
            fileuploads = Any[]
        end
    end

    @onchange active_uploaded_csv begin
        if !isempty(active_uploaded_csv)
            uploaded_files_html = Base.invokelatest(_uploaded_files_html, uploaded_csv_paths; active_path=active_uploaded_csv)
        end
    end

    @onchange btn_visualize_data begin
        if btn_visualize_data > 0
            selected_path = Base.invokelatest(_selected_data_source_path, data_source_mode, active_uploaded_csv, premade_csv_choice, csv_path_input)
            if data_source_mode == "upload"
                if isempty(uploaded_csv_paths)
                    status_msg = "<span style='color:#d97706'>&#9888; No uploaded files found. Upload one or more CSV files first.</span>"
                else
                    try
                        combined = Base.invokelatest(_combine_uploaded_data, uploaded_csv_paths)
                        if isempty(combined)
                            status_msg = "<span style='color:#d97706'>&#9888; No readable uploaded files available to visualize.</span>"
                        else
                            active_path = !isempty(active_uploaded_csv) && isfile(active_uploaded_csv) ? active_uploaded_csv : uploaded_csv_paths[1]
                            active_df = Base.invokelatest(_safe_load, active_path)
                            csv_path_loaded = active_path
                            active_uploaded_csv = active_path
                            stage_csv_select = isempty(stage_csv_select) ? active_path : (stage_csv_select in uploaded_csv_paths ? stage_csv_select : active_path)

                            data_preview_html = Base.invokelatest(_df_to_html, combined; limit=20)
                            preflight_html = Base.invokelatest(_multi_preflight_to_html, uploaded_csv_paths)
                            has_stage_metadata_flag = all([_has_stage_metadata(_safe_load(p)) for p in uploaded_csv_paths if isfile(p)])

                            conds = try
                                Base.invokelatest(build_conditions, active_df)
                            catch
                                FitCondition[]
                            end
                            if !isempty(conds)
                                conditions_list = [c.name for c in conds]
                                condition_select = conditions_list[1]
                            else
                                conditions_list = String[]
                            end

                            traces, layout = Base.invokelatest(_overview_plotdata_multi, uploaded_csv_paths)
                            overview_data = traces
                            overview_layout = layout
                            uploaded_files_html = Base.invokelatest(_uploaded_files_html, uploaded_csv_paths; active_path=active_uploaded_csv)

                            status_msg = "<span style='color:#0f766e'>&#10003; Visualized upload collection: $(length(uploaded_csv_paths)) file(s), $(nrow(combined)) total rows.</span>"
                        end
                    catch err
                        status_msg = "<span style='color:#dc2626'>&#10060; Error visualizing uploads: $(sprint(showerror, err))</span>"
                    end
                end
            elseif isempty(selected_path)
                status_msg = "<span style='color:#d97706'>&#9888; No file selected. Choose Upload, Premade, or File Path first.</span>"
            elseif !isfile(selected_path)
                status_msg = "<span style='color:#d97706'>&#9888; File not found: $(selected_path)</span>"
            else
                try
                    df = Base.invokelatest(_safe_load, selected_path)
                    csv_path_loaded = selected_path
                    uploaded_csv_paths = Base.invokelatest(_append_unique_path, copy(uploaded_csv_paths), selected_path)
                    uploaded_csv_options = Base.invokelatest(_uploaded_csv_options, uploaded_csv_paths)
                    active_uploaded_csv = selected_path
                    stage_csv_select = isempty(stage_csv_select) ? selected_path : (stage_csv_select in uploaded_csv_paths ? stage_csv_select : selected_path)
                    uploaded_files_html = Base.invokelatest(_uploaded_files_html, uploaded_csv_paths; active_path=active_uploaded_csv)

                    status_msg = "<span style='color:#0f766e'>&#10003; Visualized: $(basename(selected_path)) | $(nrow(df)) rows | $(ncol(df)) columns</span>"
                    data_preview_html = Base.invokelatest(_df_to_html, df; limit=10)
                    has_stage_metadata_flag = Base.invokelatest(_has_stage_metadata, df)
                    preflight_html = Base.invokelatest(_preflight_to_html, selected_path)

                    conds = Base.invokelatest(build_conditions, df)
                    if !isempty(conds)
                        conditions_list = [c.name for c in conds]
                        condition_select = conditions_list[1]
                    else
                        conditions_list = String[]
                    end

                    traces, layout = Base.invokelatest(_overview_plotdata_multi, uploaded_csv_paths)
                    overview_data   = traces
                    overview_layout = layout
                catch err
                    status_msg = "<span style='color:#dc2626'>&#10060; Error visualizing: $(sprint(showerror, err))</span>"
                end
            end
            btn_visualize_data = 0
        end
    end

    @onchange pipeline_stage_select begin
        if !isempty(pipeline_stages)
            idx = Base.invokelatest(_pipeline_stage_index, pipeline_stage_select, 1)
            pipeline_stages_html = Base.invokelatest(_pipeline_stage_cards_html, pipeline_stages, idx)
            pipeline_flowchart_html = Base.invokelatest(_pipeline_stage_flowchart_html, pipeline_stages)
            pipeline_mapping_html = Base.invokelatest(_pipeline_mapping_html, pipeline_stages, idx, selected_models)
            stage_dict = Dict{String,Any}(String(k) => v for (k, v) in pairs(Dict(pipeline_stages[idx])))
            stage_csv_select = String(get(stage_dict, "csv_file", ""))
            pipeline_status_html = "<p style='color:#0f766e'>Selected stage $(idx).</p>"
        end
    end

    @onchange stage_csv_select begin
        if !isempty(pipeline_stages)
            idx = Base.invokelatest(_pipeline_stage_index, pipeline_stage_select, 1)
            stages = Any[pipeline_stages...]
            stage = Dict{String,Any}(String(k) => v for (k, v) in pairs(Dict(stages[idx])))
            new_csv = String(stage_csv_select)
            old_csv = String(get(stage, "csv_file", ""))
            if new_csv != old_csv
                pipeline_status_html = "<p style='color:#0f766e'>Selected CSV for stage $(idx): $(isempty(new_csv) ? "(not set)" : basename(new_csv)). Click 'Apply CSV to Stage' to save.</p>"
            end
        end
    end

    @onchange btn_stage_apply_csv begin
        if btn_stage_apply_csv > 0
            if isempty(pipeline_stages)
                pipeline_status_html = "<p style='color:#d97706'>&#9888; Add a stage first.</p>"
            else
                idx = Base.invokelatest(_pipeline_stage_index, pipeline_stage_select, 1)
                selected_csv = strip(String(stage_csv_select))
                if isempty(selected_csv)
                    pipeline_status_html = "<p style='color:#d97706'>&#9888; Select a CSV file first.</p>"
                elseif !isfile(selected_csv)
                    pipeline_status_html = "<p style='color:#d97706'>&#9888; Selected file no longer exists: $(selected_csv)</p>"
                else
                    stages = Any[pipeline_stages...]
                    stage = Dict{String,Any}(String(k) => v for (k, v) in pairs(Dict(stages[idx])))
                    stage["csv_file"] = selected_csv
                    stages[idx] = stage
                    pipeline_stages = stages
                    pipeline_stage_options, pipeline_stage_select, pipeline_stages_html, pipeline_flowchart_html, pipeline_mapping_html =
                        Base.invokelatest(_pipeline_refresh_outputs, pipeline_stages, idx, selected_models)
                    pipeline_status_html = "<p style='color:#0f766e'>Applied CSV to stage $(idx): $(basename(selected_csv)).</p>"
                end
            end
            btn_stage_apply_csv = 0
        end
    end

    @onchange btn_add_stage begin
        if btn_add_stage > 0
            stages = Any[pipeline_stages...]
            selected_stage_path = !isempty(stage_csv_select) ? String(stage_csv_select) : ""
            current_path = !isempty(selected_stage_path) ? selected_stage_path : (isempty(csv_path_loaded) ? String(csv_path_input) : csv_path_loaded)
            push!(stages, Dict(
                "name" => "Stage $(length(stages) + 1)",
                "csv_file" => current_path,
                "model_name" => Base.invokelatest(_pipeline_default_model, selected_models),
                "param_mapping" => Dict{String,String}(),
            ))
            pipeline_stages = stages
            pipeline_stage_options, pipeline_stage_select, pipeline_stages_html, pipeline_flowchart_html, pipeline_mapping_html =
                Base.invokelatest(_pipeline_refresh_outputs, pipeline_stages, length(stages), selected_models)
            pipeline_status_html = "<p style='color:#0f766e'>Added stage $(length(stages)).</p>"
            btn_add_stage = 0
        end
    end

    @onchange btn_stage_up begin
        if btn_stage_up > 0 && !isempty(pipeline_stages)
            idx = Base.invokelatest(_pipeline_stage_index, pipeline_stage_select, 1)
            if idx > 1
                stages = Any[pipeline_stages...]
                stages[idx - 1], stages[idx] = stages[idx], stages[idx - 1]
                pipeline_stages = stages
                pipeline_stage_options, pipeline_stage_select, pipeline_stages_html, pipeline_flowchart_html, pipeline_mapping_html =
                    Base.invokelatest(_pipeline_refresh_outputs, pipeline_stages, idx - 1, selected_models)
                pipeline_status_html = "<p style='color:#0f766e'>Moved stage up.</p>"
            end
            btn_stage_up = 0
        end
    end

    @onchange btn_stage_down begin
        if btn_stage_down > 0 && !isempty(pipeline_stages)
            idx = Base.invokelatest(_pipeline_stage_index, pipeline_stage_select, 1)
            if idx < length(pipeline_stages)
                stages = Any[pipeline_stages...]
                stages[idx + 1], stages[idx] = stages[idx], stages[idx + 1]
                pipeline_stages = stages
                pipeline_stage_options, pipeline_stage_select, pipeline_stages_html, pipeline_flowchart_html, pipeline_mapping_html =
                    Base.invokelatest(_pipeline_refresh_outputs, pipeline_stages, idx + 1, selected_models)
                pipeline_status_html = "<p style='color:#0f766e'>Moved stage down.</p>"
            end
            btn_stage_down = 0
        end
    end

    @onchange btn_stage_remove begin
        if btn_stage_remove > 0 && !isempty(pipeline_stages)
            idx = Base.invokelatest(_pipeline_stage_index, pipeline_stage_select, 1)
            stages = Any[pipeline_stages...]
            deleteat!(stages, idx)
            pipeline_stages = stages
            pipeline_stage_options, pipeline_stage_select, pipeline_stages_html, pipeline_flowchart_html, pipeline_mapping_html =
                Base.invokelatest(_pipeline_refresh_outputs, pipeline_stages, isempty(stages) ? 1 : min(idx, length(stages)), selected_models)
            pipeline_status_html = "<p style='color:#0f766e'>Removed stage $(idx).</p>"
            btn_stage_remove = 0
        end
    end

    @onchange btn_stage_use_current begin
        if btn_stage_use_current > 0 && !isempty(pipeline_stages)
            idx = Base.invokelatest(_pipeline_stage_index, pipeline_stage_select, 1)
            stages = Any[pipeline_stages...]
            stage = Dict{String,Any}(String(k) => v for (k, v) in pairs(Dict(stages[idx])))
            selected_stage_path = !isempty(stage_csv_select) ? String(stage_csv_select) : ""
            stage["csv_file"] = !isempty(selected_stage_path) ? selected_stage_path : (isempty(csv_path_loaded) ? String(csv_path_input) : csv_path_loaded)
            stage["model_name"] = Base.invokelatest(_pipeline_default_model, selected_models)
            stages[idx] = stage
            pipeline_stages = stages
            pipeline_stage_options, pipeline_stage_select, pipeline_stages_html, pipeline_flowchart_html, pipeline_mapping_html =
                Base.invokelatest(_pipeline_refresh_outputs, pipeline_stages, idx, selected_models)
            pipeline_status_html = "<p style='color:#0f766e'>Copied current data and model into stage $(idx).</p>"
            btn_stage_use_current = 0
        end
    end

    @onchange btn_stage_next begin
        if btn_stage_next > 0 && !isempty(pipeline_stages)
            idx = Base.invokelatest(_pipeline_stage_index, pipeline_stage_select, 1)
            if idx < length(pipeline_stages)
                pipeline_stage_select = idx + 1
                pipeline_stage_options, pipeline_stage_select, pipeline_stages_html, pipeline_flowchart_html, pipeline_mapping_html =
                    Base.invokelatest(_pipeline_refresh_outputs, pipeline_stages, pipeline_stage_select, selected_models)
                pipeline_status_html = "<p style='color:#0f766e'>Advanced to stage $(pipeline_stage_select).</p>"
            else
                pipeline_status_html = "<p style='color:#6b7280'>The selected stage is the final stage.</p>"
            end
            btn_stage_next = 0
        end
    end

    @onchange btn_save_pipeline begin
        if btn_save_pipeline > 0
            if isempty(strip(pipeline_name))
                pipeline_status_html = "<p style='color:#dc2626'>Pipeline name cannot be empty.</p>"
            elseif isempty(pipeline_stages)
                pipeline_status_html = "<p style='color:#d97706'>Add at least one stage before saving.</p>"
            else
                try
                    stages = PipelineStage[]
                    for (i, sraw) in enumerate(pipeline_stages)
                        s = Dict{String,Any}(String(k) => v for (k, v) in pairs(Dict(sraw)))
                        push!(stages, PipelineStage(
                            String(get(s, "name", "Stage $(i)")),
                            String(get(s, "csv_file", "")),
                            String(get(s, "model_name", "")),
                            haskey(s, "param_mapping") ? Dict{String,String}(string(k) => string(v) for (k, v) in pairs(Dict(s["param_mapping"]))) : Dict{String,String}(),
                        ))
                    end
                    Base.invokelatest(_save_pipeline, Pipeline(strip(pipeline_name), stages))
                    pipeline_status_html = "<p style='color:#0f766e'>Saved pipeline '$(strip(pipeline_name))' with $(length(stages)) stage(s).</p>"
                catch err
                    pipeline_status_html = "<p style='color:#dc2626'>Save failed: $(sprint(showerror, err))</p>"
                end
            end
            btn_save_pipeline = 0
        end
    end

    @onchange selected_models begin
        model_options_list = Base.invokelatest(_model_options)
        if isempty(selected_models) || all(startswith.(selected_models, "__hdr__"))
            model_equations_html = "<p style='color:#6b7280'>Select models to see their equations.</p>"
        else
            html = "<div>"
            for m in selected_models
                startswith(m, "__hdr__") && continue
                m in Base.invokelatest(list_models) || continue
                spec   = Base.invokelatest(get_model, m)
                params = join(string.(spec.param_names), ", ")
                lat    = Base.invokelatest(_model_latex, m)
                family = Base.invokelatest(_spec_family, spec)
                html  *= "<div style='margin-bottom:12px;padding:10px;background:#f0fdf4;border-radius:6px;border-left:3px solid #0f766e'>" *
                         "<strong>$(m)</strong> <span style='color:#6b7280;font-size:12px'>($(family))</span><br>" *
                         "<span style='font-size:12px;font-family:monospace'>$(lat)</span><br>" *
                         "<span style='font-size:12px;color:#374151'>Parameters: $(params)</span></div>"
            end
            model_equations_html = html * "</div>"
        end
        if !isempty(pipeline_stages)
            idx = Base.invokelatest(_pipeline_stage_index, pipeline_stage_select, 1)
            pipeline_stage_options, pipeline_stage_select, pipeline_stages_html, pipeline_flowchart_html, pipeline_mapping_html =
                Base.invokelatest(_pipeline_refresh_outputs, pipeline_stages, idx, selected_models)
        end
    end

    @onchange builder_equations begin
        try
            snames = Base.invokelatest(_parse_state_names, builder_state_names)
            eq_map = Base.invokelatest(_parse_equation_lines, builder_equations, snames)
            pname  = isempty(strip(builder_model_name)) ? "custom_model" : strip(builder_model_name)
            plist  = [strip(p) for p in split(builder_param_names, ",") if !isempty(strip(p))]
            obs    = isempty(strip(builder_observable)) ? string(first(snames)) : strip(builder_observable)
            eq_lines = join(["d$(s)/dt = $(eq_map[s])" for s in snames], "<br>")
            builder_preview_html = "<div style='background:#f0fdf4;border:1px solid #d1fae5;padding:12px;border-radius:6px;font-size:13px'>" *
                "<strong>$(pname)</strong><br><strong>Dynamics:</strong><br>$(eq_lines)<br>" *
                "<strong>Observable:</strong> y(t) = $(obs)<br><strong>Parameters:</strong> $(join(plist, ", "))</div>"
        catch err
            builder_preview_html = "<p style='color:#dc2626;font-size:12px'>Preview error: $(sprint(showerror, err))</p>"
        end
    end

    @onchange btn_load_template begin
        if btn_load_template > 0
            tmpl = Base.invokelatest(_builder_template_data, builder_template)
            builder_family        = tmpl.family
            builder_state_names   = tmpl.states
            builder_observable    = tmpl.observable
            builder_param_names   = tmpl.params
            builder_constants     = tmpl.constants
            builder_lower_bounds  = tmpl.lower
            builder_upper_bounds  = tmpl.upper
            builder_equations     = tmpl.equations
            btn_load_template = 0
        end
    end

    @onchange btn_register_model begin
        if btn_register_model > 0
            try
                spec = Base.invokelatest(_build_custom_model_spec,
                    builder_model_name, builder_family, builder_state_names,
                    builder_observable, builder_param_names, builder_constants,
                    builder_lower_bounds, builder_upper_bounds, builder_equations)
                Base.invokelatest(register_model!, spec; overwrite=true)
                Base.invokelatest(_save_gui_model_to_file,
                    builder_model_name, builder_family, builder_state_names,
                    builder_observable, builder_param_names, builder_constants,
                    builder_lower_bounds, builder_upper_bounds, builder_equations)
                custom_model_status = "<small style='color:#0f766e'>&#10003; Model '$(builder_model_name)' registered and saved.</small>"
                model_options_list  = Base.invokelatest(_model_options)
            catch err
                custom_model_status = "<small style='color:#dc2626'>&#10060; Registration failed: $(sprint(showerror, err))</small>"
            end
            btn_register_model = 0
        end
    end

    @onchange btn_run_fit begin
        if btn_run_fit > 0
            valid_models = [String(m) for m in selected_models if !startswith(m, "__hdr__")]
            if isempty(csv_path_loaded)
                rank_html = "<div style='background:#fffbeb;border:1px solid #fcd34d;padding:10px;border-radius:6px'>&#9888; Load a dataset first (Tab 1).</div>"
            elseif isempty(valid_models)
                rank_html = "<div style='background:#fffbeb;border:1px solid #fcd34d;padding:10px;border-radius:6px'>&#9888; Select at least one model (Tab 4).</div>"
            else
                try
                    panel_html, traces, layout = Base.invokelatest(_rank_output_html,
                        csv_path_loaded, valid_models, condition_select,
                        n_starts, maxiters, train_frac, split_mode_v, uncertainty_boot)
                    rank_html   = panel_html
                    fit_data    = traces
                    fit_layout  = layout
                catch err
                    rank_html = "<div style='background:#fef2f2;border:1px solid #fca5a5;padding:10px;border-radius:6px'>&#10060; Fit failed: $(sprint(showerror, err))</div>"
                end
            end
            btn_run_fit = 0
        end
    end

    @onchange btn_run_pipeline begin
        if btn_run_pipeline > 0
            if isempty(csv_path_loaded)
                pipeline_output_html = "<div style='background:#fffbeb;border:1px solid #fcd34d;padding:10px;border-radius:6px'>&#9888; Load a dataset first (Tab 1).</div>"
            else
                valid_models = isempty(selected_models) ? Base.invokelatest(_default_models) :
                    [String(m) for m in selected_models if !startswith(m, "__hdr__")]
                try
                    pipeline_output_html = Base.invokelatest(_pipeline_output_html, csv_path_loaded, valid_models)
                catch err
                    pipeline_output_html = "<div style='background:#fef2f2;border:1px solid #fca5a5;padding:10px;border-radius:6px'>&#10060; Pipeline failed: $(sprint(showerror, err))</div>"
                end
            end
            btn_run_pipeline = 0
        end
    end

    @onchange btn_run_staged begin
        if btn_run_staged > 0
            if isempty(csv_path_loaded)
                staged_html = "<div style='background:#fffbeb;border:1px solid #fcd34d;padding:10px;border-radius:6px'>&#9888; Load a dataset first (Tab 1).</div>"
            elseif isempty(pipeline_stages) && !has_stage_metadata_flag
                staged_html = "<div style='background:#fffbeb;border:1px solid #fcd34d;padding:10px;border-radius:6px'>&#9888; Dataset lacks stage metadata. Load 'Staged monoculture' or 'Staged co-culture' from Tab 1.</div>"
            else
                try
                    staged_html = Base.invokelatest(_staged_output_html, csv_path_loaded; pipeline_stages=pipeline_stages)
                catch err
                    staged_html = "<div style='background:#fef2f2;border:1px solid #fca5a5;padding:10px;border-radius:6px'>&#10060; Staged pipeline failed: $(sprint(showerror, err))</div>"
                end
            end
            btn_run_staged = 0
        end
    end

end

# ══════════════════════════════════════════════════════════════════════════════
# UI layout (Stipple/Quasar components)
# ══════════════════════════════════════════════════════════════════════════════

const _SPLIT_OPTS = [
    Dict("label" => "Temporal (early -> train, late -> validation)", "value" => "temporal"),
    Dict("label" => "Random (seeded at 42)", "value" => "random"),
]

const _FAMILY_OPTS = [Dict("label" => s, "value" => s) for s in
    ["custom", "logistic", "gompertz", "theta_logistic", "coculture", "mechanistic"]]

const _TEMPLATE_OPTS = [
    Dict("label" => "Single-state logistic",           "value" => "logistic"),
    Dict("label" => "Single-state Gompertz",           "value" => "gompertz"),
    Dict("label" => "Theta-logistic + Hill inhibition","value" => "theta_hill"),
    Dict("label" => "Sensitive / resistant (2-state)", "value" => "sensitive_resistant"),
    Dict("label" => "Lotka-Volterra competition",      "value" => "lotka_volterra"),
]

const _DATA_SOURCE_OPTS = [
    Dict("label" => "Upload", "value" => "upload"),
    Dict("label" => "Premade", "value" => "premade"),
    Dict("label" => "File Path", "value" => "filepath"),
]

const _PREMADE_CSV_OPTS = [
    Dict("label" => "Basic Pipeline", "value" => "basic_pipeline"),
    Dict("label" => "Staged Monoculture", "value" => "staged_monoculture"),
    Dict("label" => "Staged Co-culture", "value" => "coculture_stages"),
]

function ui()
    [
        Genie.Renderer.Html.div(class="q-pa-md", [

            # ── Header ────────────────────────────────────────────────────────
            h2("GrowthParameterEstimation", style="color:#0f766e;margin:0"),
            p("Fit, compare, and rank ODE growth models against your cell-count time series.",
              style="color:#6b7280;margin:4px 0 12px 0"),
            Genie.Renderer.Html.div([]; v__html = "status_msg",
                style="background:#f0fdf4;border:1px solid #d1fae5;padding:10px 14px;border-radius:6px;margin-bottom:16px"),

            # ── Tab navigation ────────────────────────────────────────────────
            tabgroup(:active_tab, align="left", indicator__color="teal", active__color="teal",
                     class="bg-white shadow-2", style="margin-bottom:0", [
                tab(name="tab-load",   label="1. Load Data"),
                tab(name="tab-pipeline-design", label="2. Pipeline Designer"),
                tab(name="tab-build",  label="3. Build Models"),
                tab(name="tab-models", label="4. Select Models"),
                tab(name="tab-rank",   label="5. Fit & Rank"),
                tab(name="tab-staged", label="6. Staged Analysis"),
            ]),
            separator(),

            # ── Tab panels ────────────────────────────────────────────────────
            tabpanelgroup(:active_tab, animated=true, [

                # ──── Tab 1: Load Data ─────────────────────────────────────────
                tabpanel(name="tab-load", class="q-pa-md", [
                    h5("Load CSV Dataset", style="color:#0f766e;margin:0 0 8px 0"),
                                        p("Choose a source (Upload, Premade, or File Path), then click Visualize Data when you are ready.",
                      style="font-size:13px;color:#6b7280;margin-bottom:12px"),
                    uploader(
                        accept = ".csv",
                        autoupload = true,
                        hideuploadbtn = true,
                        multiple = true,
                        nothumbnails = true,
                        label = "Drag and drop one or more CSV files here",
                        style = "max-width: 100%; width: 100%; margin-bottom: 16px;"
                    ),
                    StippleUI.Selects.select(:data_source_mode, options=_DATA_SOURCE_OPTS,
                        label="Data source", outlined=true,
                        hint="Pick where the next visualization should come from.", class="q-mb-md"),
                    Genie.Renderer.Html.div([
                        StippleUI.Selects.select(:active_uploaded_csv, options=:uploaded_csv_options,
                            label="Uploaded file", outlined=true,
                            hint="Choose one uploaded file as the active dataset for downstream fit/pipeline actions.", class="q-mb-md"),
                    ]; v__show="data_source_mode === 'upload'"),
                    Genie.Renderer.Html.div([
                        StippleUI.Selects.select(:premade_csv_choice, options=_PREMADE_CSV_OPTS,
                            label="Premade dataset", outlined=true,
                            hint="Used when Data source is Premade.", class="q-mb-md"),
                    ]; v__show="data_source_mode === 'premade'"),
                    Genie.Renderer.Html.div([
                        textfield("CSV file path", :csv_path_input,
                            placeholder="C:\\Users\\...\\data.csv  or  /home/.../data.csv",
                            hint="Used when Data source is File Path.",
                            outlined=true, class="q-mb-md"),
                    ]; v__show="data_source_mode === 'filepath'"),
                    Genie.Renderer.Html.div(class="q-gutter-sm q-mb-md", [
                        btn("Visualize Data", color="teal", icon="visibility", @click("btn_visualize_data += 1")),
                    ]),
                    separator(class="q-mb-md"),
                    h6("Data Preview (first 10 rows)", style="color:#374151;margin:0 0 6px 0"),
                    Genie.Renderer.Html.div([]; v__html = "data_preview_html"),
                    h6("Uploaded File Summary", style="color:#374151;margin:16px 0 6px 0"),
                    Genie.Renderer.Html.div([]; v__html = "uploaded_files_html"),
                    h6("Raw Trajectory Overview", style="color:#374151;margin:16px 0 6px 0"),
                    plot(:overview_data, layout=:overview_layout, id="overview-plot"),
                    h6("Preflight Quality Checks", style="color:#374151;margin:16px 0 6px 0"),
                    Genie.Renderer.Html.div([]; v__html = "preflight_html"),
                ]),

                # ──── Tab 2: Pipeline Designer ─────────────────────────────────
                tabpanel(name="tab-pipeline-design", class="q-pa-md", [
                    h5("Design stage order and data flow", style="color:#0f766e;margin:0 0 8px 0"),
                    p("Define each stage in order. Then move to Tab 3/4/5 for model building, per-stage model selection, and fitting.",
                      style="font-size:13px;color:#6b7280;margin-bottom:12px"),
                    Genie.Renderer.Html.div(class="row q-col-gutter-md", [
                        Genie.Renderer.Html.div(class="col-8", [
                            Genie.Renderer.Html.div(class="row q-col-gutter-md q-mb-sm", [
                                Genie.Renderer.Html.div(class="col-8", [
                                    textfield("Pipeline name", :pipeline_name, value="my_pipeline",
                                        hint="e.g. untreated_to_treated", outlined=true),
                                ]),
                                Genie.Renderer.Html.div(class="col-4 q-pt-sm", [
                                    btn("Add Stage", color="teal", outline=true, @click("btn_add_stage += 1")),
                                    btn("Save Pipeline", color="teal", outline=true, @click("btn_save_pipeline += 1")),
                                ]),
                            ]),
                            Genie.Renderer.Html.div(class="row q-col-gutter-md q-mb-sm", [
                                Genie.Renderer.Html.div(class="col-8", [
                                    StippleUI.Selects.select(:pipeline_stage_select, options=:pipeline_stage_options,
                                        label="Selected stage", outlined=true),
                                ]),
                                Genie.Renderer.Html.div(class="col-4 q-pt-sm", [
                                    btn("Move Up", color="teal", outline=true, @click("btn_stage_up += 1")),
                                    btn("Move Down", color="teal", outline=true, @click("btn_stage_down += 1")),
                                    btn("Remove", color="teal", outline=true, @click("btn_stage_remove += 1")),
                                ]),
                            ]),
                            Genie.Renderer.Html.div(class="row q-col-gutter-md q-mb-sm", [
                                Genie.Renderer.Html.div(class="col-12", [
                                    StippleUI.Selects.select(:stage_csv_select, options=:uploaded_csv_options,
                                        label="Stage CSV file", outlined=true,
                                        hint="Choose from uploaded files. Then click Apply CSV to Stage.", class="q-mb-sm"),
                                    btn("Apply CSV to Stage", color="teal", outline=true, @click("btn_stage_apply_csv += 1")),
                                ]),
                            ]),
                            Genie.Renderer.Html.div(class="q-gutter-sm q-mb-md", [
                                btn("Use Current Data+Model", color="teal", outline=true, @click("btn_stage_use_current += 1")),
                                btn("→ Next Stage", color="teal", outline=true, @click("btn_stage_next += 1")),
                            ]),
                        ]),
                        Genie.Renderer.Html.div(class="col-4", [
                            Genie.Renderer.Html.div([]; v__html = "pipeline_status_html"),
                        ]),
                    ]),
                    Genie.Renderer.Html.div(class="q-mb-md", [
                        Genie.Renderer.Html.div([]; v__html = "pipeline_stages_html"),
                    ]),
                    h6("Pipeline flow", style="color:#374151;margin:16px 0 6px 0"),
                    Genie.Renderer.Html.div([];
                        v__html = "pipeline_flowchart_html",
                        id="pipeline-flowchart",
                        class="q-mb-md",
                        style="border:1px dashed #d1fae5;border-radius:6px;padding:12px;background:#f9fafb"),
                    h6("Carry-over mapping", style="color:#374151;margin:16px 0 6px 0"),
                    Genie.Renderer.Html.div([]; v__html = "pipeline_mapping_html"),
                ]),

                # ──── Tab 3: Build Models ──────────────────────────────────────
                tabpanel(name="tab-build", class="q-pa-md", [
                    h5("GUI Model Builder", style="color:#0f766e;margin:0 0 8px 0"),
                    p("Define a custom ODE growth model with state equations, parameters, and bounds. No coding required.",
                      style="font-size:13px;color:#6b7280;margin-bottom:12px"),
                    Genie.Renderer.Html.div(class="row q-col-gutter-md q-mb-md", [
                        Genie.Renderer.Html.div(class="col", [
                            StippleUI.Selects.select(:builder_template, options=_TEMPLATE_OPTS,
                                label="Start from template", outlined=true),
                        ]),
                        Genie.Renderer.Html.div(class="col-auto q-pt-sm", [
                            btn("Load Template", color="teal", outline=true,
                                @click("btn_load_template += 1")),
                        ]),
                    ]),
                    Genie.Renderer.Html.div(class="row q-col-gutter-md q-mb-sm", [
                        Genie.Renderer.Html.div(class="col", [ textfield("Model name", :builder_model_name, outlined=true) ]),
                        Genie.Renderer.Html.div(class="col", [
                            StippleUI.Selects.select(:builder_family, options=_FAMILY_OPTS,
                                label="Family label", outlined=true),
                        ]),
                    ]),
                    Genie.Renderer.Html.div(class="row q-col-gutter-md q-mb-sm", [
                        Genie.Renderer.Html.div(class="col", [
                            textfield("State names (comma-separated)", :builder_state_names,
                                hint="e.g.  N  or  S, R", outlined=true),
                        ]),
                        Genie.Renderer.Html.div(class="col", [
                            textfield("Observable expression", :builder_observable,
                                hint="e.g.  N  or  S + R", outlined=true),
                        ]),
                    ]),
                    Genie.Renderer.Html.div(class="row q-col-gutter-md q-mb-sm", [
                        Genie.Renderer.Html.div(class="col", [
                            textfield("Parameter names (comma-separated)", :builder_param_names,
                                hint="e.g.  r, K", outlined=true),
                        ]),
                        Genie.Renderer.Html.div(class="col", [
                            textfield("Preset constants (name=value, ...)", :builder_constants,
                                hint="e.g.  hill=1.0, ic50=0.5  (optional)", outlined=true),
                        ]),
                    ]),
                    Genie.Renderer.Html.div(class="row q-col-gutter-md q-mb-md", [
                        Genie.Renderer.Html.div(class="col", [
                            textfield("Lower bounds (comma-separated)", :builder_lower_bounds,
                                hint="e.g.  1e-6, 1e-3", outlined=true),
                        ]),
                        Genie.Renderer.Html.div(class="col", [
                            textfield("Upper bounds (comma-separated)", :builder_upper_bounds,
                                hint="e.g.  5.0, 1e7", outlined=true),
                        ]),
                    ]),
                    StippleUI.textarea("State equations  (one per line:  state = expression)", :builder_equations,
                        hint="Available: state names, parameter names, t (time), E (exposure/dose)",
                        rows=5, outlined=true, class="q-mb-md"),
                    h6("Live Preview", style="color:#374151;margin:0 0 4px 0"),
                    Genie.Renderer.Html.div([]; v__html = "builder_preview_html", class="q-mb-md"),
                    btn("Register Model", color="teal", icon="save",
                        @click("btn_register_model += 1")),
                    Genie.Renderer.Html.div([]; v__html = "custom_model_status", class="q-mt-sm"),
                ]),

                # ──── Tab 4: Select Models ─────────────────────────────────────
                tabpanel(name="tab-models", class="q-pa-md", [
                    h5("Model Selection", style="color:#0f766e;margin:0 0 8px 0"),
                    p("Select one or more models to compare. Use Tab 3 to add custom models.",
                      style="font-size:13px;color:#6b7280;margin-bottom:12px"),
                    StippleUI.Selects.select(:selected_models, options=:model_options_list,
                        multiple=true, use__chips=true, label="Models to fit",
                        hint="Built-in catalog + any custom models registered in Tab 3.",
                        outlined=true, class="q-mb-md"),
                    Genie.Renderer.Html.div(class="row q-col-gutter-md q-mb-md", [
                        Genie.Renderer.Html.div(class="col", [
                            numberfield("Optimization starts", :n_starts,
                                min=1, max=200, outlined=true,
                                hint="More starts = better quality, slower"),
                        ]),
                        Genie.Renderer.Html.div(class="col", [
                            numberfield("Max iterations per start", :maxiters,
                                min=20, max=10000, outlined=true,
                                hint="Increase for complex models"),
                        ]),
                    ]),
                    h6("Selected Model Reference", style="color:#374151;margin:0 0 4px 0"),
                    Genie.Renderer.Html.div([]; v__html = "model_equations_html"),
                ]),

                # ──── Tab 5: Fit & Rank ────────────────────────────────────────
                tabpanel(name="tab-rank", class="q-pa-md", [
                    h5("Fit & Rank Models", style="color:#0f766e;margin:0 0 8px 0"),
                    p("Loads current dataset (Tab 1) and selected models (Tab 4). Performs train/validation split, multi-start fitting, ranking, and bootstrap uncertainty.",
                      style="font-size:13px;color:#6b7280;margin-bottom:12px"),
                    Genie.Renderer.Html.div(class="row q-col-gutter-md q-mb-sm", [
                        Genie.Renderer.Html.div(class="col", [
                            textfield("Condition to plot", :condition_select,
                                hint="Leave blank to use the first condition",
                                outlined=true),
                        ]),
                        Genie.Renderer.Html.div(class="col", [
                            numberfield("Train fraction", :train_frac,
                                min=0.5, max=0.95, step=0.05, outlined=true,
                                hint="Fraction of timepoints used for training (0.5-0.95)"),
                        ]),
                    ]),
                    Genie.Renderer.Html.div(class="row q-col-gutter-md q-mb-md", [
                        Genie.Renderer.Html.div(class="col", [
                            StippleUI.Selects.select(:split_mode_v, options=_SPLIT_OPTS,
                                label="Split mode", outlined=true),
                        ]),
                        Genie.Renderer.Html.div(class="col", [
                            numberfield("Bootstrap samples (95% CI)", :uncertainty_boot,
                                min=5, max=200, outlined=true),
                        ]),
                    ]),
                    Genie.Renderer.Html.div(class="q-gutter-sm q-mb-lg", [
                        btn("Run Fit & Rank", color="teal", icon="play_arrow",
                            @click("btn_run_fit += 1")),
                        btn("Run Full Pipeline", color="teal", outline=true, icon="rocket_launch",
                            @click("btn_run_pipeline += 1")),
                    ]),
                    h6("Model vs Data", style="color:#374151;margin:0 0 4px 0"),
                    plot(:fit_data, layout=:fit_layout, id="fit-plot"),
                    separator(class="q-my-md"),
                    h6("Ranking Results", style="color:#374151;margin:0 0 4px 0"),
                    Genie.Renderer.Html.div([]; v__html = "rank_html"),
                    h6("Full Pipeline Output", style="color:#374151;margin:16px 0 4px 0"),
                    Genie.Renderer.Html.div([]; v__html = "pipeline_output_html"),
                ]),

                # ──── Tab 6: Staged Analysis ───────────────────────────────────
                tabpanel(name="tab-staged", class="q-pa-md", [
                    h5("Complete Staged Analysis", style="color:#0f766e;margin:0 0 8px 0"),
                    p("Load a dataset with stage metadata (culture_type and population_type columns) in Tab 1, then run the staged pipeline here.",
                      style="font-size:13px;color:#6b7280;margin-bottom:12px"),
                    Genie.Renderer.Html.div(class="q-mb-lg", [
                        btn("Run Staged Analysis", color="teal", icon="science",
                            @click("btn_run_staged += 1")),
                    ]),
                    Genie.Renderer.Html.div([]; v__html = "staged_html"),
                ]),

            ]),  # end tabpanelgroup
        ]),  # end root div
    ]
end

# ══════════════════════════════════════════════════════════════════════════════
# Mount route and launch
# ══════════════════════════════════════════════════════════════════════════════

@page("/", ui, model=GrowthGUI)

println("GrowthParameterEstimation GUI (Genie/Stipple) ready at http://$(HOST):$(PORT)")
up(PORT, async=false)
