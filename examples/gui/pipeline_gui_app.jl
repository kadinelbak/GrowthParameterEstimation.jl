using Dash
using CSV
using DataFrames
using GrowthParameterEstimation
using Statistics
using Base64
using TOML
using Random

# Force Julia recompilation by changing a comment (v3)

const HOST = "127.0.0.1"
const PORT = parse(Int, get(ENV, "GPE_GUI_PORT", "8050"))
const LATEX_CONFIG_PATH = joinpath(dirname(dirname(dirname(@__FILE__))), "config", "model_latex.toml")
const EXAMPLE_DIR = joinpath(dirname(@__FILE__), "data")
const GUI_CUSTOM_MODELS_PATH = joinpath(EXAMPLE_DIR, "gui_custom_models.toml")
const GUI_PIPELINES_PATH = joinpath(EXAMPLE_DIR, "gui_pipelines.toml")

# ── Pipeline data structures ──────────────────────────────────────────────────
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

# ── LaTeX config ──────────────────────────────────────────────────────────────
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

# ── Model catalog (dynamic to allow runtime custom registration) ──────────────

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

function _model_options(models::Vector{String}=list_models())
    by_family = Dict{String, Vector{String}}()
    for m in models
        f = get_model(m).base_growth_family
        push!(get!(by_family, f, String[]), m)
    end
    options = Any[]
    for family in FAMILY_SORT_ORDER
        models = get(by_family, family, String[])
        isempty(models) && continue
        label = get(FAMILY_LABELS, family, family)
        push!(options, Dict("label" => "── $(label) ──", "value" => "__hdr__$(family)", "disabled" => true))
        for m in sort(models)
            push!(options, Dict("label" => m, "value" => m))
        end
    end
    return options
end

function _default_models()
    available = list_models()
    preferred = ["logistic_growth", "gompertz_growth"]
    chosen = [m for m in preferred if m in available]
    if isempty(chosen)
        return isempty(available) ? String[] : available[1:min(end, 2)]
    end
    return chosen
end

# ── Helpers ───────────────────────────────────────────────────────────────────

function _model_latex(model_name::AbstractString)
    m = String(model_name)
    if haskey(LATEX_CACHE, m)
        return LATEX_CACHE[m]
    end

    spec = get_model(m)
    ptxt = join(string.(spec.param_names), ", ")
    fam = lowercase(spec.base_growth_family)

    if fam == "logistic"
        return """
**$(m)**

\$\$
\\frac{dN}{dt} = rN\\left(1-\\frac{N}{K}\\right)
\$\$

Parameters: `$(ptxt)`
"""
    elseif fam == "gompertz"
        return """
**$(m)**

\$\$
\\frac{dN}{dt} = rN\\ln\\left(\\frac{K}{N}\\right)
\$\$

Parameters: `$(ptxt)`
"""
    elseif fam == "theta_logistic"
        return """
**$(m)**

\$\$
\\frac{dN}{dt} = rN\\left(1-\\left(\\frac{N}{K}\\right)^{\\theta}\\right) - g(E(t),\\theta)N
\$\$

Parameters: `$(ptxt)`
"""
    else
        return """
**$(m)**

\$\$
\\dot{\\mathbf{u}}(t) = f\\!\\left(\\mathbf{u}(t), t; \\theta, E(t)\\right)
\$\$

\$\$
y(t) = h\\!\\left(\\mathbf{u}(t)\\right)
\$\$

Parameters: `$(ptxt)`
"""
    end
end

function _safe_load(path::AbstractString)
    raw = CSV.read(path, DataFrame)
    return normalize_schema(raw)
end

function _has_stage_metadata(df::DataFrame)
    cols = Set(Symbol.(names(df)))
    return (:culture_type in cols) && (:population_type in cols)
end

function _condition_cols_for_plot(df::DataFrame)
    cols = Set(Symbol.(names(df)))
    preferred = [:dose, :cell_line, :density, :replicate]
    found = [c for c in preferred if c in cols]
    return found
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

function _overview_traces(plot_df::DataFrame; normalize::Bool = false)
    traces = Any[]
    grouped = groupby(plot_df, :condition_label)
    for g in grouped
        g_sorted = sort(g, :time)
        y = Float64.(g_sorted.count)
        if normalize
            y0 = max(1e-12, y[1])
            y = y ./ y0
        end
        push!(traces, Dict(
            "x" => Float64.(g_sorted.time),
            "y" => y,
            "mode" => "lines+markers",
            "name" => String(g_sorted.condition_label[1]),
            "line" => Dict("width" => 2),
            "marker" => Dict("size" => 7),
        ))
    end
    return traces
end

function _data_overview_panels(df::DataFrame)
    plot_df = _plot_df_with_condition(df)
    traces_raw = _overview_traces(plot_df; normalize=false)
    traces_norm = _overview_traces(plot_df; normalize=true)

    fig_raw = Dict("data" => traces_raw, "layout" => Dict(
        "title" => "Raw trajectories by condition",
        "xaxis" => Dict("title" => "Time"),
        "yaxis" => Dict("title" => "Observed count"),
        "hovermode" => "closest",
        "plot_bgcolor" => "#f9fafb",
        "paper_bgcolor" => "#ffffff",
    ))

    fig_log = Dict("data" => traces_raw, "layout" => Dict(
        "title" => "Raw trajectories (log-scale y)",
        "xaxis" => Dict("title" => "Time"),
        "yaxis" => Dict("title" => "Observed count", "type" => "log"),
        "hovermode" => "closest",
        "plot_bgcolor" => "#f9fafb",
        "paper_bgcolor" => "#ffffff",
    ))

    fig_norm = Dict("data" => traces_norm, "layout" => Dict(
        "title" => "Normalized trajectories (count / first count in condition)",
        "xaxis" => Dict("title" => "Time"),
        "yaxis" => Dict("title" => "Relative count"),
        "hovermode" => "closest",
        "plot_bgcolor" => "#f9fafb",
        "paper_bgcolor" => "#ffffff",
    ))

    grouped = groupby(plot_df, :condition_label)
    conds = String[]
    endpoints = Float64[]
    aucs = Float64[]
    for g in grouped
        g_sorted = sort(g, :time)
        t = Float64.(g_sorted.time)
        y = Float64.(g_sorted.count)
        push!(conds, String(g_sorted.condition_label[1]))
        push!(endpoints, y[end])
        area = 0.0
        for i in 2:length(t)
            dt = t[i] - t[i - 1]
            area += dt * (y[i] + y[i - 1]) / 2
        end
        push!(aucs, area)
    end

    fig_end = Dict("data" => Any[
        Dict("x" => conds, "y" => endpoints, "type" => "bar", "name" => "Final count", "marker" => Dict("color" => "#0f766e")),
        Dict("x" => conds, "y" => aucs, "type" => "bar", "name" => "AUC", "marker" => Dict("color" => "#0891b2")),
    ], "layout" => Dict(
        "title" => "Condition-level outcome summary",
        "xaxis" => Dict("title" => "Condition", "tickangle" => -25),
        "yaxis" => Dict("title" => "Value"),
        "barmode" => "group",
        "plot_bgcolor" => "#f9fafb",
        "paper_bgcolor" => "#ffffff",
    ))

    return _card([
        _help("These plots show exactly what will be fitted: raw trajectories, scale-sensitive view, normalized dynamics, and condition-level outcomes."),
        dcc_graph(figure=fig_raw;  style=Dict("height" => "340px")),
        dcc_graph(figure=fig_log;  style=Dict("height" => "340px")),
        dcc_graph(figure=fig_norm; style=Dict("height" => "340px")),
        dcc_graph(figure=fig_end;  style=Dict("height" => "360px")),
    ]; title="Data To Be Fitted — Visual Overview")
end

function _keybtn(label::AbstractString, id::AbstractString)
    html_button(label; id=id, n_clicks=0, style=Dict(
        "background" => "#ecfeff", "color" => "#155e75", "border" => "1px solid #a5f3fc",
        "padding" => "6px 10px", "borderRadius" => "6px", "cursor" => "pointer",
        "fontWeight" => "600", "fontSize" => "12px", "marginRight" => "8px", "marginBottom" => "8px"))
end

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
    if isnothing(text) || isempty(strip(String(text)))
        return Symbol[], Float64[]
    end
    names = Symbol[]
    values = Float64[]
    items = [strip(part) for part in split(String(text), ",") if !isempty(strip(part))]
    for item in items
        occursin("=", item) || throw(ArgumentError("Preset constants must use `name=value` format, got: $(item)"))
        lhs, rhs = split(item, "="; limit=2)
        cname = strip(lhs)
        cval = strip(rhs)
        isempty(cname) && throw(ArgumentError("Constant name cannot be empty in `$(item)`"))
        push!(names, Symbol(cname))
        push!(values, parse(Float64, cval))
    end
    return names, values
end

function _validate_rhs_expr(expr, allowed_symbols::Set{Symbol})
    if expr isa Number
        return nothing
    elseif expr isa Symbol
        expr in allowed_symbols || throw(ArgumentError("Unsupported symbol in equation: $(expr)"))
        return nothing
    elseif expr isa Expr
        if expr.head == :call
            fn = expr.args[1]
            allowed_calls = Set([:+, :-, :*, :/, :^, :log, :exp, :sqrt, :sin, :cos, :tan, :abs, :max, :min])
            (fn isa Symbol && fn in allowed_calls) || throw(ArgumentError("Unsupported function/operator in equation: $(fn)"))
            for arg in expr.args[2:end]
                _validate_rhs_expr(arg, allowed_symbols)
            end
            return nothing
        end
        throw(ArgumentError("Unsupported expression form in equation: $(expr.head)"))
    end
    throw(ArgumentError("Unsupported equation element: $(expr)"))
end

function _rhs_to_tex(rhs::AbstractString)
    s = replace(String(rhs), "log(" => "\\log(")
    s = replace(s, "exp(" => "\\exp(")
    s = replace(s, "sqrt(" => "\\sqrt(")
    s = replace(s, "*" => " ")
    s = replace(s, "(" => "\\left(")
    s = replace(s, ")" => "\\right)")
    return s
end

function _parse_equation_lines(text, state_names::Vector{Symbol})
    isnothing(text) && throw(ArgumentError("State equations are required"))
    lines = [strip(line) for line in split(String(text), '\n') if !isempty(strip(line))]
    isempty(lines) && throw(ArgumentError("State equations cannot be empty"))

    eq_map = Dict{Symbol,String}()
    for line in lines
        occursin("=", line) || throw(ArgumentError("Each equation line must have the form `state = expression`"))
        lhs, rhs = split(line, "="; limit=2)
        state = Symbol(strip(lhs))
        state in state_names || throw(ArgumentError("Equation provided for unknown state: $(state)"))
        eq_map[state] = strip(rhs)
    end

    for state in state_names
        haskey(eq_map, state) || throw(ArgumentError("Missing equation for state $(state)"))
    end
    return eq_map
end

function _compile_expression(expr_text::AbstractString, state_names::Vector{Symbol}, param_names::Vector{Symbol}, constant_names::Vector{Symbol}=Symbol[])
    parsed = try
        Meta.parse(String(expr_text))
    catch err
        throw(ArgumentError("Could not parse expression `$(expr_text)`: $(sprint(showerror, err))"))
    end
    allowed = Set(vcat(state_names, [:t, :E, :pi], param_names, constant_names))
    _validate_rhs_expr(parsed, allowed)
    fn_expr = Expr(:->, Expr(:tuple, state_names..., :t, :E, param_names..., constant_names...), parsed)
    return Base.eval(@__MODULE__, fn_expr)
end

function _builder_template_data(template::AbstractString)
    if template == "sensitive_resistant"
        return (
            family="mechanistic",
            states="S, R",
            observable="S + R",
            params="rS, rR, K, kSR, emax, ic50, hill",
            constants="",
            lower="1e-6, 1e-6, 1e-3, 0.0, 0.0, 1e-8, 0.1",
            upper="5.0, 5.0, 1e7, 2.0, 20.0, 1e4, 8.0",
            equations="S = rS*S*(1 - (S + R)/K) - kSR*S - emax*(E^hill/(ic50^hill + E^hill))*S\nR = rR*R*(1 - (S + R)/K) + kSR*S",
        )
    elseif template == "lotka_volterra"
        return (
            family="coculture",
            states="S, R",
            observable="S + R",
            params="rS, KS, alphaSR, rR, KR, alphaRS",
            constants="",
            lower="1e-6, 1e-3, 0.0, 1e-6, 1e-3, 0.0",
            upper="5.0, 1e7, 5.0, 5.0, 1e7, 5.0",
            equations="S = rS*S*(1 - (S + alphaSR*R)/KS)\nR = rR*R*(1 - (R + alphaRS*S)/KR)",
        )
    elseif template == "theta_hill"
        return (
            family="theta_logistic",
            states="N",
            observable="N",
            params="r, K, theta, ic50, hill",
            constants="",
            lower="1e-6, 1e-3, 0.1, 1e-8, 0.1",
            upper="5.0, 1e7, 5.0, 1e4, 8.0",
            equations="N = r*N*(1 - (N/K)^theta) * (1 - E^hill/(ic50^hill + E^hill))",
        )
    elseif template == "gompertz"
        return (
            family="gompertz",
            states="N",
            observable="N",
            params="a, b, K",
            constants="",
            lower="1e-6, 1e-6, 1e-3",
            upper="5.0, 10.0, 1e7",
            equations="N = a*N*log(K/N)",
        )
    end

    return (
        family="logistic",
        states="N",
        observable="N",
        params="r, K",
        constants="",
        lower="1e-6, 1e-3",
        upper="5.0, 1e7",
        equations="N = r*N*(1 - N/K)",
    )
end

function _custom_model_preview(name, state_text, observable_text, params_text, constants_text, equations_text)
    if isnothing(equations_text) || isempty(strip(String(equations_text)))
        return html_p("Enter state equations to preview the model in math text.", style=Dict("color" => "#6b7280"))
    end

    states = _parse_state_names(state_text)
    eq_map = _parse_equation_lines(equations_text, states)
    params = isnothing(params_text) ? String[] : [strip(p) for p in split(String(params_text), ",") if !isempty(strip(p))]
    constants = isnothing(constants_text) ? String[] : [strip(c) for c in split(String(constants_text), ",") if !isempty(strip(c))]
    pname = isnothing(name) || isempty(strip(String(name))) ? "custom_model" : strip(String(name))
    eq_blocks = join(["\\frac{d$(state)}{dt} = $(_rhs_to_tex(eq_map[state]))" for state in states], "\\\\\n")
    obs_tex = isnothing(observable_text) || isempty(strip(String(observable_text))) ? first(states) : strip(String(observable_text))
    latex = """
**$(pname)**

\$\$
$(eq_blocks)
\$\$

\$\$
y(t) = $(_rhs_to_tex(obs_tex))
\$\$

Parameters: `$(join(params, ", "))`

Preset constants: `$(join(constants, ", "))`

Use state names from your state list, `t` for time, and `E` for exposure/dose.
"""
    return dcc_markdown(latex, mathjax=true)
end

function _save_gui_model_to_file(name, family, states, observable, params, constants, lower, upper, equations)
    data = isfile(GUI_CUSTOM_MODELS_PATH) ? TOML.parsefile(GUI_CUSTOM_MODELS_PATH) : Dict{String,Any}()
    if !haskey(data, "models")
        data["models"] = Dict{String,Any}()
    end
    data["models"][name] = Dict{String,Any}(
        "family"    => String(family),
        "states"    => String(states),
        "observable"=> String(observable),
        "params"    => String(params),
        "constants" => String(constants),
        "lower"     => String(lower),
        "upper"     => String(upper),
        "equations" => String(equations),
    )
    open(GUI_CUSTOM_MODELS_PATH, "w") do io
        TOML.print(io, data)
    end
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
                get(mdata, "name", mname),
                get(mdata, "family", "custom"),
                get(mdata, "states", "N"),
                get(mdata, "observable", "N"),
                get(mdata, "params", "r, K"),
                get(mdata, "constants", ""),
                get(mdata, "lower", "1e-6, 1e-3"),
                get(mdata, "upper", "5.0, 1e7"),
                get(mdata, "equations", "N = 0.0"),
            )
            register_model!(spec; overwrite=true)
            n_loaded += 1
        catch err
            @warn "Could not restore GUI model '$(mname)': $(sprint(showerror, err))"
        end
    end
    n_loaded > 0 && @info "Restored $(n_loaded) GUI-built model(s) from $(GUI_CUSTOM_MODELS_PATH)"
end

function _build_custom_model_spec(name, family, state_text, observable_text, params_text, constants_text, lower_text, upper_text, equations_text)
    model_name = strip(String(name))
    isempty(model_name) && throw(ArgumentError("Model name is required"))

    state_names = _parse_state_names(state_text)
    param_names = _parse_symbol_csv(params_text, "Parameter names")
    constant_names, constant_values = _parse_constants_csv(constants_text)
    lower_bounds = _parse_float_csv(lower_text, "Lower bounds")
    upper_bounds = _parse_float_csv(upper_text, "Upper bounds")
    overlap = intersect(Set(param_names), Set(constant_names))
    isempty(overlap) || throw(ArgumentError("Symbols cannot be both fitted parameters and preset constants: $(join(string.(collect(overlap)), ", "))"))
    length(param_names) == length(lower_bounds) == length(upper_bounds) ||
        throw(ArgumentError("Parameter names, lower bounds, and upper bounds must have the same length"))

    eq_map = _parse_equation_lines(equations_text, state_names)
    rhs_fns = Dict(state => _compile_expression(eq_map[state], state_names, param_names, constant_names) for state in state_names)

    observable_expr = isnothing(observable_text) || isempty(strip(String(observable_text))) ? string(first(state_names)) : strip(String(observable_text))
    observable_fn = _compile_expression(observable_expr, state_names, param_names, constant_names)

    ode_fn = function (du, u, p, t, exposure)
        E = exposure(t)
        state_values = Tuple(u[i] for i in eachindex(state_names))
        for (idx, state) in enumerate(state_names)
            du[idx] = rhs_fns[state](state_values..., t, E, p..., constant_values...)
        end
        return nothing
    end

    bounds = [(lower_bounds[i], upper_bounds[i]) for i in eachindex(param_names)]
    fam = isnothing(family) || isempty(strip(String(family))) ? "custom" : strip(String(family))

    return ModelSpec(
        name=model_name,
        ode! = ode_fn,
        param_names=param_names,
        bounds=bounds,
        n_states=length(state_names),
        observable=u -> observable_fn(Tuple(u[i] for i in eachindex(state_names))..., 0.0, 0.0, zeros(length(param_names))..., constant_values...),
        base_growth_family=fam,
        state_names=state_names,
        metadata=Dict(:source => :gui_builder, :equation_rhs => join(["$(s)=$(eq_map[s])" for s in state_names], "; "), :observable_expr => observable_expr, :preset_constants => Dict(string(constant_names[i]) => constant_values[i] for i in eachindex(constant_names))),
    )
end

function _as_table(df::DataFrame; limit::Int = 15)
    isempty(df) && return html_p("(empty)", style=Dict("color" => "#6b7280"))
    shown = nrow(df) > limit ? first(df, limit) : df
    header = html_tr([html_th(string(c); style=Dict("padding" => "4px 8px", "background" => "#f0fdf4", "fontWeight" => "600", "fontSize" => "12px")) for c in names(shown)])
    rows = [html_tr([html_td(string(shown[i, c]); style=Dict("padding" => "3px 8px", "fontSize" => "12px")) for c in names(shown)]) for i in 1:nrow(shown)]
    return html_div([
        html_table([html_thead(header), html_tbody(rows)]; style=Dict("width" => "100%", "borderCollapse" => "collapse", "border" => "1px solid #e5e7eb")),
        nrow(df) > limit ? html_small("Showing $(nrow(shown)) of $(nrow(df)) rows", style=Dict("color" => "#6b7280")) : nothing,
    ])
end

function _condition_key(name::AbstractString)
    lowercase(strip(String(name)))
end

function _find_condition(conditions::Vector{FitCondition}, name::AbstractString)
    key = _condition_key(name)
    findfirst(c -> _condition_key(c.name) == key, conditions)
end

function _save_upload(contents::AbstractString, filename::AbstractString)
    _, b64 = split(contents, ","; limit=2)
    data = base64decode(b64)
    tmpdir = mktempdir()
    path = joinpath(tmpdir, filename)
    write(path, data)
    return path
end

function _triggered_id()
    ctx = callback_context()
    isempty(ctx.triggered) && return nothing
    first(split(String(ctx.triggered[1].prop_id), "."))
end

# ── Styled helper components ──────────────────────────────────────────────────

function _card(children; title=nothing)
    header_el = isnothing(title) ? nothing : html_div(title; style=Dict(
        "background" => "#0f766e", "color" => "#fff", "fontWeight" => "bold",
        "padding" => "8px 16px", "fontSize" => "14px", "borderRadius" => "6px 6px 0 0",
    ))
    body_el = html_div(children; style=Dict(
        "padding" => "16px", "border" => "1px solid #d1fae5",
        "borderTop"    => isnothing(title) ? "1px solid #d1fae5" : "none",
        "borderRadius" => isnothing(title) ? "6px" : "0 0 6px 6px",
    ))
    return html_div([header_el, body_el]; style=Dict("marginBottom" => "20px", "borderRadius" => "6px", "boxShadow" => "0 1px 4px rgba(0,0,0,0.08)"))
end

function _help(text::AbstractString)
    html_p(text; style=Dict("color" => "#6b7280", "fontSize" => "12px", "margin" => "4px 0 8px 0"))
end

function _alert(text::AbstractString; kind=:info)
    color  = kind == :error ? "#fef2f2" : kind == :warn ? "#fffbeb" : "#f0fdf4"
    border = kind == :error ? "#fca5a5" : kind == :warn ? "#fcd34d" : "#6ee7b7"
    html_div(text; style=Dict("background" => color, "border" => "1px solid $(border)",
        "padding" => "10px 14px", "borderRadius" => "6px", "fontSize" => "13px", "marginBottom" => "12px"))
end

function _btn(label, id; n_clicks=0)
    html_button(label; id=id, n_clicks=n_clicks, style=Dict(
        "background" => "#0f766e", "color" => "#fff", "border" => "none",
        "padding" => "10px 22px", "borderRadius" => "6px", "cursor" => "pointer",
        "fontWeight" => "600", "fontSize" => "14px", "marginRight" => "10px"))
end

# ── Pipeline persistence ──────────────────────────────────────────────────────

function _save_pipeline(pipeline::Pipeline)
    config = if isfile(GUI_PIPELINES_PATH)
        try
            TOML.parsefile(GUI_PIPELINES_PATH)
        catch
            Dict{String, Any}()
        end
    else
        Dict{String, Any}()
    end

    pipelines = haskey(config, "pipelines") ? Dict(config["pipelines"]) : Dict{String, Any}()
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
end

function _load_pipelines()::Dict{String, Pipeline}
    if !isfile(GUI_PIPELINES_PATH)
        return Dict{String, Pipeline}()
    end
    
    try
        config = TOML.parsefile(GUI_PIPELINES_PATH)
        result = Dict{String, Pipeline}()
        
        if haskey(config, "pipelines")
            for (name, pline) in config["pipelines"]
                stages = []
                if haskey(pline, "stages")
                    for stage_dict in pline["stages"]
                        stage = PipelineStage(
                            stage_dict["name"],
                            stage_dict["csv_file"],
                            stage_dict["model_name"],
                            Dict(stage_dict["param_mapping"])
                        )
                        push!(stages, stage)
                    end
                end
                result[name] = Pipeline(name, stages)
            end
        end
        return result
    catch
        return Dict{String, Pipeline}()
    end
end

function _glossary()
    html_div([
        html_div([
            html_button("📖 Glossary"; id="glossary-toggle", n_clicks=0,
                style=Dict(
                    "background" => "transparent", "border" => "1px solid #a7f3d0",
                    "color" => "#0f766e", "borderRadius" => "20px", "padding" => "4px 14px",
                    "fontSize" => "12px", "fontWeight" => "600", "cursor" => "pointer",
                    "lineHeight" => "1.4")),
        ]; style=Dict("position" => "absolute", "top" => "18px", "right" => "24px")),
        html_div(id="glossary-panel",
            children=html_div([
                html_dl([
                    html_dt("Condition", style=Dict("fontWeight" => "600")),
                    html_dd("One unique combination of experimental factors (dose, cell line, replicate …). Each condition is fitted independently."),

                    html_dt("BIC  (Bayesian Information Criterion)", style=Dict("fontWeight" => "600", "marginTop" => "8px")),
                    html_dd("A score that balances goodness-of-fit against model complexity. Lower is better. Use BIC to compare models with different numbers of parameters."),

                    html_dt("SSR  (Sum of Squared Residuals)", style=Dict("fontWeight" => "600", "marginTop" => "8px")),
                    html_dd("Raw fit quality: sum of squared differences between observed data and model prediction. Lower is better, but does not penalise extra parameters."),

                    html_dt("r  —  growth rate", style=Dict("fontWeight" => "600", "marginTop" => "8px")),
                    html_dd("Intrinsic per-capita growth rate. Larger r → faster initial growth."),

                    html_dt("K  —  carrying capacity", style=Dict("fontWeight" => "600", "marginTop" => "8px")),
                    html_dd("Maximum sustainable population size under logistic constraints."),

                    html_dt("IC50", style=Dict("fontWeight" => "600", "marginTop" => "8px")),
                    html_dd("Drug concentration that produces 50 % of the maximum effect. Smaller IC50 → cells are more drug-sensitive."),

                    html_dt("Hill coefficient  (h)", style=Dict("fontWeight" => "600", "marginTop" => "8px")),
                    html_dd("Controls the steepness of the dose-response curve. h = 1 is hyperbolic; h > 1 gives a sharper switch."),

                    html_dt("Preflight check", style=Dict("fontWeight" => "600", "marginTop" => "8px")),
                    html_dd("Automatic data-quality scan that flags missing columns, sparse conditions, duplicate timepoints, or outliers before fitting."),
                ]),
            ]; style=Dict("padding" => "12px 0 4px 0")),
            style=Dict("display" => "none", "border" => "1px solid #d1fae5", "borderRadius" => "8px",
                       "padding" => "12px 18px", "marginTop" => "8px", "background" => "#f0fdf4",
                       "boxShadow" => "0 2px 8px rgba(0,0,0,0.07)")),
    ]; style=Dict())
end

# ── Tab 1 helpers ─────────────────────────────────────────────────────────────

function _preflight_output(path::AbstractString)
    df       = _safe_load(path)
    preflight = preflight_data_quality(df)
    n_cond   = length(build_conditions(df))
    staged   = _has_stage_metadata(df)

    issue_items = [
        html_li(r.severity * ": " * r.message;
            style=Dict("color" => (r.severity == "error" ? "#dc2626" : "#d97706"), "marginBottom" => "4px"))
        for r in eachrow(preflight.issues)
    ]

    return _card([
        html_h5("Data Quality Summary", style=Dict("marginTop" => "0")),
        _as_table(preflight.summary; limit=20),
        html_h5("Condition Quality"),
        _as_table(preflight.condition_quality; limit=30),
        isempty(preflight.issues) ?
            _alert("No data quality issues found. ✓") :
            html_div([html_h5("Issues"), html_ul(issue_items)]),
    ]; title="Preflight Report — $(basename(path))  ($(n_cond) conditions$(staged ? ", staged" : ""))")
end

# ── Tab 3 helpers ─────────────────────────────────────────────────────────────

function _split_indices(n::Int, train_fraction::Float64; mode::Symbol = :temporal, seed::Int = 42)
    n_train = clamp(round(Int, train_fraction * n), 2, n - 2)
    if mode == :random
        rng = MersenneTwister(seed)
        idx = shuffle(rng, collect(1:n))
        train_idx = sort(idx[1:n_train])
        val_idx = sort(idx[(n_train + 1):end])
    else
        train_idx = collect(1:n_train)
        val_idx = collect((n_train + 1):n)
    end
    return train_idx, val_idx
end

function _slice_condition(cond::FitCondition, idx::Vector{Int})
    return FitCondition(
        cond.name,
        cond.time[idx],
        cond.count[idx],
        cond.error[idx],
        [max(1e-12, cond.count[idx[1]])],
        cond.exposure,
        cond.metadata,
    )
end

function _rmse(y::Vector{Float64}, yhat::Vector{Float64})
    return sqrt(mean((y .- yhat) .^ 2))
end

function _mape(y::Vector{Float64}, yhat::Vector{Float64})
    denom = max.(abs.(y), 1e-8)
    return 100.0 * mean(abs.((y .- yhat) ./ denom))
end

function _fit_dose_for_condition(cond::FitCondition)
    return evaluate_exposure(cond.exposure, cond.time[1])
end

function _bootstrap_interval(
    spec::ModelSpec,
    cond_train::FitCondition,
    cond_full::FitCondition;
    n_boot::Int = 30,
    maxiters::Int = 150,
)
    n = length(cond_train.time)
    n < 4 && return nothing

    rng = MersenneTwister(42)
    preds = Vector{Vector{Float64}}()
    dose = _fit_dose_for_condition(cond_train)

    for _ in 1:n_boot
        idx = rand(rng, 1:n, n)
        xb = cond_train.time[idx]
        yb = cond_train.count[idx]
        order = sortperm(xb)
        xb = xb[order]
        yb = yb[order]

        try
            fit_res = fit_model(spec, xb, yb, dose; maxiters=maxiters)
            sim = simulate(
                spec,
                cond_full.time,
                fit_res.params;
                u0=cond_full.u0,
                exposure=cond_full.exposure,
            )
            if sim.success && length(sim.observed) == length(cond_full.time) && all(isfinite, sim.observed)
                push!(preds, Float64.(sim.observed))
            end
        catch
        end
    end

    length(preds) < 5 && return nothing
    mat = hcat(preds...)
    lower = [quantile(vec(mat[i, :]), 0.025) for i in 1:size(mat, 1)]
    upper = [quantile(vec(mat[i, :]), 0.975) for i in 1:size(mat, 1)]

    return (lower=lower, upper=upper, n_success=length(preds))
end

function _presentation_template(
    dataset_name::String,
    split_mode::Symbol,
    train_fraction::Float64,
    best_model::String,
    best_train_bic::Float64,
    best_val_rmse::Float64,
    has_interval::Bool,
    interval_samples::Int,
)
    protocol = split_mode == :random ?
        "Random split by condition (seed=42)" :
        "Temporal split by condition (early timepoints train, late timepoints validation)"

    md = """
### Standard Presentation Template

#### 1) Data + protocol
- Dataset: **$(dataset_name)**
- Split protocol: **$(protocol)**
- Train fraction: **$(round(train_fraction * 100, digits=1))%**

#### 2) Model selection
- Selected model: **$(best_model)**
- Training BIC: **$(round(best_train_bic, digits=4))**
- Validation RMSE: **$(isfinite(best_val_rmse) ? string(round(best_val_rmse, digits=4)) : "not available")**

#### 3) Uncertainty
- 95% predictive interval: **$(has_interval ? "available" : "not available")**
- Bootstrap fits retained: **$(interval_samples)**

#### 4) Recommendation statement
"We selected **$(best_model)** because it achieved the best validation performance under a fixed train/validation protocol while maintaining interpretable parameterization for growth dynamics."
"""

    return _card([
        dcc_markdown(md, mathjax=true),
    ]; title="Final Report (Standardized Template)")
end

function _rank_output(path, models, cond_name, n_starts, maxiters, train_fraction, split_mode, n_boot)
    df         = _safe_load(path)
    conditions = build_conditions(df)
    isempty(conditions) && return (
        _alert("No conditions built. Check preflight issues on the Load Data tab.", kind=:warn),
        Dict("data" => Any[], "layout" => Dict("title" => "No conditions available")))
    isempty(models) && (models = _default_models())

    split_fraction = clamp(Float64(train_fraction), 0.5, 0.95)
    split_sym = split_mode == "random" ? :random : :temporal

    split_rows = NamedTuple[]
    train_conditions = FitCondition[]
    val_conditions = Dict{String,FitCondition}()
    full_conditions = Dict{String,FitCondition}()

    for cond in conditions
        full_conditions[cond.name] = cond
        n_pts = length(cond.time)
        if n_pts < 4
            push!(train_conditions, cond)
            push!(split_rows, (condition=cond.name, n_total=n_pts, n_train=n_pts, n_val=0, split_applied=false))
            continue
        end

        train_idx, val_idx = _split_indices(n_pts, split_fraction; mode=split_sym, seed=42)
        c_train = _slice_condition(cond, train_idx)
        c_val = _slice_condition(cond, val_idx)
        push!(train_conditions, c_train)
        val_conditions[cond.name] = c_val
        push!(split_rows, (condition=cond.name, n_total=n_pts, n_train=length(train_idx), n_val=length(val_idx), split_applied=true))
    end

    split_df = DataFrame(split_rows)

    ranked = rank_models(models, train_conditions;
        n_starts=n_starts, maxiters=maxiters, top_k=min(length(models), 5), seed=42)

    val_rows = NamedTuple[]
    for model_name in ranked.ranking.model
        if !haskey(ranked.fits, model_name)
            push!(val_rows, (model=model_name, val_rmse=Inf, val_mape=Inf, val_sse=Inf, val_points=0))
            continue
        end

        spec = get_model(model_name)
        fi = ranked.fits[model_name]
        val_obs = Float64[]
        val_pred = Float64[]

        for (cond_name_key, c_val) in pairs(val_conditions)
            hit = findfirst(pc -> _condition_key(pc.condition) == _condition_key(cond_name_key) && pc.success, fi.per_condition)
            isnothing(hit) && continue

            p = fi.per_condition[hit].params
            sim = simulate(spec, c_val.time, p; u0=c_val.u0, exposure=c_val.exposure)
            if sim.success && length(sim.observed) == length(c_val.count)
                append!(val_obs, c_val.count)
                append!(val_pred, Float64.(sim.observed))
            end
        end

        if isempty(val_obs)
            push!(val_rows, (model=model_name, val_rmse=Inf, val_mape=Inf, val_sse=Inf, val_points=0))
        else
            sse = sum((val_obs .- val_pred) .^ 2)
            push!(val_rows, (
                model=model_name,
                val_rmse=_rmse(val_obs, val_pred),
                val_mape=_mape(val_obs, val_pred),
                val_sse=sse,
                val_points=length(val_obs),
            ))
        end
    end

    val_df = DataFrame(val_rows)
    sort!(val_df, :val_rmse)
    ranking_table = leftjoin(ranked.ranking, val_df; on=:model)
    sort!(ranking_table, [:val_rmse, :bic])

    # Plot for selected (or first) condition
    selected_cond = isempty(cond_name) ? conditions[1].name : cond_name
    cond_idx = _find_condition(conditions, selected_cond)
    isnothing(cond_idx) && (cond_idx = 1; selected_cond = conditions[1].name)
    cond = conditions[cond_idx]

    traces = Any[Dict("x" => cond.time, "y" => cond.count, "mode" => "markers",
        "name" => "Observed", "marker" => Dict("size" => 9, "color" => "#111827"))]

    for model_name in ranking_table.model
        haskey(ranked.fits, model_name) || continue
        fi  = ranked.fits[model_name]
        hit = findfirst(pc -> _condition_key(pc.condition) == _condition_key(selected_cond) && pc.success, fi.per_condition)
        isnothing(hit) && continue

        spec = get_model(model_name)
        p = fi.per_condition[hit].params
        sim = simulate(spec, cond.time, p; u0=cond.u0, exposure=cond.exposure)
        sim.success || continue

        push!(traces, Dict("x" => cond.time, "y" => Float64.(sim.observed),
            "mode" => "lines", "name" => model_name, "line" => Dict("width" => 2)))
    end

    interval_success = 0
    if !isempty(ranking_table)
        best_model = String(ranking_table.model[1])
        if haskey(ranked.fits, best_model) && haskey(val_conditions, selected_cond)
            spec = get_model(best_model)
            selected_train = findfirst(c -> _condition_key(c.name) == _condition_key(selected_cond), train_conditions)
            if !isnothing(selected_train)
                ci = _bootstrap_interval(
                    spec,
                    train_conditions[selected_train],
                    cond;
                    n_boot=max(Int(n_boot), 5),
                    maxiters=min(maxiters, 250),
                )
                if !isnothing(ci)
                    interval_success = ci.n_success
                    push!(traces, Dict(
                        "x" => vcat(cond.time, reverse(cond.time)),
                        "y" => vcat(ci.upper, reverse(ci.lower)),
                        "fill" => "toself",
                        "fillcolor" => "rgba(15,118,110,0.15)",
                        "line" => Dict("color" => "rgba(15,118,110,0.05)"),
                        "name" => "95% interval ($(best_model))",
                        "hoverinfo" => "skip",
                    ))
                end
            end
        end
    end

    fig = Dict("data" => traces, "layout" => Dict(
        "title"       => "Condition: $(selected_cond)",
        "xaxis"       => Dict("title" => "Time"),
        "yaxis"       => Dict("title" => "Count"),
        "legend"      => Dict("orientation" => "v"),
        "hovermode"   => "closest",
        "plot_bgcolor"  => "#f9fafb",
        "paper_bgcolor" => "#ffffff",
    ))

    best_model_name = isempty(ranking_table) ? "none" : String(ranking_table.model[1])
    best_train_bic = isempty(ranking_table) ? Inf : Float64(ranking_table.bic[1])
    best_val_rmse = isempty(ranking_table) ? Inf : Float64(ranking_table.val_rmse[1])

    template_card = _presentation_template(
        basename(path),
        split_sym,
        split_fraction,
        best_model_name,
        best_train_bic,
        best_val_rmse,
        interval_success > 0,
        interval_success,
    )

    panel = _card([
        html_h5("Model Ranking (validation-first)", style=Dict("marginTop" => "0")),
        _help("Selection is ordered by validation RMSE first, then training BIC. This prevents overfitting and gives an explicit train/validation protocol."),
        _as_table(ranking_table; limit=20),
        html_h5("Train/Validation Split Summary"),
        _as_table(split_df; limit=30),
        html_h5("Uncertainty"),
        html_p(interval_success > 0 ?
            "95% predictive interval displayed (bootstrap fits retained: $(interval_success))." :
            "No predictive interval available for this condition/model (insufficient successful bootstrap fits)."),
        template_card,
    ]; title="Ranking Results")

    return panel, fig
end

# ── Tab 4 helper ──────────────────────────────────────────────────────────────

function _pipeline_output(path, models)
    df     = _safe_load(path)
    isempty(models) && (models = _default_models())
    cfg    = default_config(output_dir="results/gui_pipeline")
    run    = run_pipeline(df; config=cfg, include_models=models,
        strict_schema=false, qc_before_fit=true, preflight_before_fit=true)
    return _card([
        _alert("Pipeline complete.  Ranked models: $(nrow(run.ranking))  |  Failures: $(nrow(run.failures))"),
        html_h5("Full Model Ranking", style=Dict("marginTop" => "0")),
        _as_table(run.ranking; limit=30),
        nrow(run.failures) > 0 ? html_div([html_h5("Failures"), _as_table(run.failures; limit=20)]) : nothing,
    ]; title="Full Pipeline Results")
end

# ── Tab 5 helpers (staged) ────────────────────────────────────────────────────

function _staged_condition_best_models(run)
    rows = NamedTuple[]
    for stage in run.stages
        (stage.status != "completed" || isnothing(stage.result)) && continue
        for (model_name, fi) in pairs(stage.result.fits)
            for pc in fi.per_condition
                pc.success || continue
                push!(rows, (stage=String(stage.name), condition=String(pc.condition), model=String(model_name)))
            end
        end
    end
    isempty(rows) && return DataFrame(stage=String[], condition=String[], best_model=String[])
    return DataFrame(rows)
end

function _staged_ranking_table(run)
    rows = NamedTuple[]
    for stage in run.stages
        (stage.status != "completed" || isnothing(stage.result)) && continue
        for r in eachrow(stage.result.ranking)
            push!(rows, (stage=String(stage.name), model=String(r.model), bic=Float64(r.bic)))
        end
    end
    isempty(rows) && return DataFrame(stage=String[], model=String[], bic=Float64[])
    return DataFrame(rows)
end

function _staged_figure_for_stage(run, stage_name::AbstractString)
    overlay_path = nothing
    for stage in run.stages
        String(stage.name) != stage_name && continue
        (stage.status != "completed" || isnothing(stage.output_dir)) && break
        fig_dir  = joinpath(String(stage.output_dir), "figures")
        isdir(fig_dir) || break
        overlays = filter(p -> endswith(p, "_overlay.csv"), readdir(fig_dir; join=true))
        isempty(overlays) || (overlay_path = first(sort(overlays)))
        break
    end
    isnothing(overlay_path) && return Dict("data" => Any[], "layout" => Dict("title" => "No overlay for stage: $(stage_name)"))

    ov = CSV.read(overlay_path, DataFrame)
    (:time in Symbol.(names(ov)) && :observed in Symbol.(names(ov))) ||
        return Dict("data" => Any[], "layout" => Dict("title" => "Overlay CSV missing required columns"))

    traces = Any[Dict("x" => ov.time, "y" => ov.observed, "mode" => "markers",
        "name" => "Observed", "marker" => Dict("size" => 8, "color" => "#111827"))]
    for c in sort([c for c in names(ov) if startswith(String(c), "pred_")])
        push!(traces, Dict("x" => ov.time, "y" => ov[!, c], "mode" => "lines",
            "name" => replace(String(c), "pred_" => ""), "line" => Dict("width" => 2)))
    end
    return Dict("data" => traces, "layout" => Dict(
        "title"         => "Stage: $(stage_name)",
        "xaxis"         => Dict("title" => "Time"),
        "yaxis"         => Dict("title" => "Count"),
        "hovermode"     => "closest",
        "plot_bgcolor"  => "#f9fafb",
        "paper_bgcolor" => "#ffffff",
    ))
end

function _staged_output(path)
    df = _safe_load(path)
    if !_has_stage_metadata(df)
        return _alert("This dataset does not have staged-pipeline metadata (culture_type and population_type columns are required). Load one of the 'Staged' example datasets from Tab 1 to try this workflow.", kind=:warn)
    end

    cfg = default_config(output_dir="results/gui_staged")
    run = run_staged_pipeline(df; stages=default_stages(), config=cfg, selection_mode=:best_bic,
        strict_schema=false, qc_before_fit=true, preflight_before_fit=true, export_stage_results=true)

    stage_rows = DataFrame(
        stage=[s.name for s in run.stages],
        status=[s.status == "skipped" ? "not_applicable" : s.status for s in run.stages],
        n_conditions=[s.n_conditions for s in run.stages],
    )

    summary_card = _card([
        html_h5("Stage Status", style=Dict("marginTop" => "0")),
        _as_table(stage_rows; limit=20),
        html_h5("Best Model By Condition"),
        _as_table(_staged_condition_best_models(run); limit=100),
        html_h5("Ranking By Stage"),
        _as_table(_staged_ranking_table(run); limit=100),
    ]; title="Staged Pipeline Summary")

    stage_cards = Any[]
    for s in run.stages
        stage_name   = String(s.name)
        stage_status = s.status == "skipped" ? "not_applicable" : String(s.status)
        color = s.status == "completed" ? "#0f766e" : "#6b7280"
        body  = if s.status == "completed"
            fig = _staged_figure_for_stage(run, stage_name)
            html_div([dcc_graph(figure=fig; style=Dict("height" => "400px"))])
        else
            html_p("Status: $(stage_status) — no plot available."; style=Dict("padding" => "12px", "color" => "#6b7280"))
        end
        push!(stage_cards, html_div([
            html_div("Stage: $(uppercase(stage_name))  [$(stage_status)]"; style=Dict(
                "background" => color, "color" => "#fff", "fontWeight" => "bold",
                "padding" => "8px 16px", "fontSize" => "14px", "borderRadius" => "6px 6px 0 0")),
            html_div(body; style=Dict("border" => "1px solid #d1fae5", "borderTop" => "none",
                "padding" => "12px", "borderRadius" => "0 0 6px 6px")),
        ]; style=Dict("marginBottom" => "24px")))
    end

    halt_msg = isnothing(run.halted_stage) ? "none" : String(run.halted_stage)
    return html_div([
        _alert("Staged run complete.  Stages completed: $(run.completed)  |  Halted at: $(halt_msg)"),
        summary_card,
        stage_cards...,
        nrow(run.failures) > 0 ? _card([html_h5("Failures"), _as_table(run.failures; limit=20)]) : nothing,
    ])
end

# ══════════════════════════════════════════════════════════════════════════════
# Startup: restore any GUI-built models that were saved in previous sessions
# ══════════════════════════════════════════════════════════════════════════════
_load_gui_models_from_file()

# ══════════════════════════════════════════════════════════════════════════════
# App layout
# ══════════════════════════════════════════════════════════════════════════════

app = dash(
    external_stylesheets=["https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css"],
    external_scripts=["https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"],
)

_tab_style    = Dict("padding" => "10px 18px", "fontWeight" => "600", "color" => "#374151")
_tab_selected = Dict("padding" => "10px 18px", "fontWeight" => "700", "color" => "#0f766e",
                     "borderBottom" => "3px solid #0f766e")

app.layout = html_div([

    # ── Hidden stores ─────────────────────────────────────────────────────────
    dcc_store(id="csv-path-store"),
    dcc_store(id="conditions-store"),
    dcc_store(id="status-bar"),

    # ── Header ────────────────────────────────────────────────────────────────
    html_header([
        html_div([
            html_h2("GrowthParameterEstimation",
                style=Dict("margin" => "0", "color" => "#0f766e")),
            html_p("Fit, compare, and rank ODE growth models against your cell-count time series.",
                style=Dict("margin" => "4px 0 0 0", "color" => "#6b7280")),
        ]),
        _glossary(),
    ]; style=Dict("position" => "relative", "borderBottom" => "2px solid #d1fae5",
                  "marginBottom" => "20px", "paddingBottom" => "12px")),

    # ── Tabs ──────────────────────────────────────────────────────────────────
    dcc_tabs(id="main-tabs", value="tab-load", children=[

        # ── Tab 1: Load Data ──────────────────────────────────────────────────
        dcc_tab(label="1. Load Data", value="tab-load",
            style=_tab_style, selected_style=_tab_selected, children=[
            html_div([
                html_br(),
                _card([
                    html_h5("Option A — Drag & drop your CSV file", style=Dict("marginTop" => "0")),
                    dcc_upload(
                        id="upload-data",
                        children=html_div(["Drag and drop, or ", html_a("click to select a file")]),
                        style=Dict(
                            "width" => "100%", "height" => "80px", "lineHeight" => "80px",
                            "borderWidth" => "2px", "borderStyle" => "dashed",
                            "borderRadius" => "8px", "borderColor" => "#6ee7b7",
                            "textAlign" => "center", "cursor" => "pointer",
                            "color" => "#0f766e", "background" => "#f0fdf4"),
                        multiple=false),
                    html_br(),
                    html_h5("Option B — Load a built-in example dataset"),
                    _help("Use these to explore the app before loading your own data."),
                    html_div([
                        _btn("Basic pipeline (2 conditions)", "btn-ex-basic"),
                        _btn("Staged monoculture", "btn-ex-staged-mono"),
                        _btn("Staged co-culture", "btn-ex-staged-co"),
                    ]),
                    html_br(),
                    html_h5("Option C — Enter a file path (for developers)"),
                    _help("Absolute path to a CSV file on this machine."),
                    dcc_input(id="manual-path", type="text", debounce=true,
                        placeholder="/path/to/your/data.csv",
                        style=Dict("width" => "100%", "fontFamily" => "monospace", "fontSize" => "13px")),
                ]; title="Load CSV Data"),

                _card([
                    html_div(id="load-data-preview",
                        children=html_p("Load a dataset above to preview the raw data that will be fitted.",
                            style=Dict("color" => "#6b7280"))),
                ]; title="Data Preview"),

                dcc_loading(id="load-spinner", type="circle", color="#0f766e",
                    children=html_div(id="load-tab-output",
                        children=html_p("Load a dataset above to see preflight quality checks here.",
                            style=Dict("color" => "#6b7280")))),
            ])
        ]),

        # ── Tab 2: Pipeline Designer ─────────────────────────────────────────
        dcc_tab(label="2. Pipeline Designer", value="tab-pipeline-design",
            style=_tab_style, selected_style=_tab_selected, children=[
            html_div([
                html_br(),
                _card([
                    html_h5("Design stage order and data flow", style=Dict("marginTop" => "0")),
                    _help("Define each stage in order. Then move to Tab 3/4/5 for model building, per-stage model selection, and fitting."),
                    html_div([
                        html_div([
                            html_label("Pipeline name"),
                            dcc_input(id="pipeline-name", type="text", value="my_pipeline",
                                placeholder="e.g., untreated_to_treated",
                                style=Dict("width" => "100%")),
                        ]),
                        html_div([
                            html_label("Actions"),
                            html_div([
                                _btn("Add Stage", "btn-add-stage"),
                                _btn("Save Pipeline", "btn-save-pipeline"),
                            ]),
                        ]),
                    ]; style=Dict("display" => "grid", "gridTemplateColumns" => "2fr 1fr", "gap" => "16px", "alignItems" => "end")),
                    html_br(),
                    html_div([
                        html_div([
                            html_label("Selected stage"),
                            dcc_dropdown(id="pipeline-stage-select", options=[], value=nothing,
                                placeholder="Add stages first…", style=Dict("fontSize" => "13px")),
                        ]),
                        html_div([
                            html_label("Order / remove"),
                            html_div([
                                _btn("Move Up", "btn-stage-up"),
                                _btn("Move Down", "btn-stage-down"),
                                _btn("Remove", "btn-stage-remove"),
                                _btn("Use Current Data+Model", "btn-stage-use-current"),
                            ]),
                        ]),
                        html_div([
                            html_label("Stage progression"),
                            html_div([
                                _btn("→ Next Stage", "btn-stage-next"),
                            ]),
                        ]),
                    ]; style=Dict("display" => "grid", "gridTemplateColumns" => "2fr 1fr", "gap" => "16px", "alignItems" => "end")),
                ]; title="Pipeline Setup"),

                dcc_loading(id="pipeline-load-spinner", type="circle", color="#0f766e",
                    children=html_div(id="pipeline-stages-container",
                        children=html_p("Click 'Add Stage' to define stage order and CSV mapping.", style=Dict("color" => "#6b7280")))),

                _card([
                    html_h5("Pipeline flow", style=Dict("marginTop" => "0")),
                    html_div(id="pipeline-flowchart",
                        children=html_p("Stage 1 → Stage 2 → Stage 3 (configured order shown here)",
                            style=Dict("color" => "#6b7280", "textAlign" => "center", "padding" => "40px")),
                        style=Dict("border" => "1px dashed #d1fae5", "borderRadius" => "6px", "padding" => "20px")),
                ]; title="Pipeline Overview"),

                dcc_store(id="pipeline-stages-store", data=Dict("stages" => [])),
                dcc_store(id="pipeline-current-stage", data=1),
            ])
        ]),

        # ── Tab 3: Build Models ───────────────────────────────────────────────
        dcc_tab(label="3. Build Models", value="tab-build",
            style=_tab_style, selected_style=_tab_selected, children=[
            html_div([
                html_br(),
                _card([
                    html_h5("Build a custom ODE model directly in the browser", style=Dict("marginTop" => "0")),
                    _help("Define your own growth model using state equations, parameters, and bounds. No coding required."),
                    html_div([
                        html_div([
                            html_label("Start from template / block set"),
                            dcc_dropdown(
                                id="builder-template",
                                options=[
                                    Dict("label" => "Single-state logistic", "value" => "logistic"),
                                    Dict("label" => "Single-state Gompertz", "value" => "gompertz"),
                                    Dict("label" => "Theta-logistic + Hill inhibition", "value" => "theta_hill"),
                                    Dict("label" => "Sensitive / resistant", "value" => "sensitive_resistant"),
                                    Dict("label" => "Lotka-Volterra competition", "value" => "lotka_volterra"),
                                ],
                                value="logistic",
                                clearable=false,
                                style=Dict("fontSize" => "13px"),
                            ),
                        ]),
                        html_div([
                            html_label("Load template"),
                            _btn("Load into builder", "btn-load-builder-template"),
                        ]),
                    ]; style=Dict("display" => "grid", "gridTemplateColumns" => "2fr 1fr", "gap" => "16px", "alignItems" => "end")),
                    html_br(),
                    html_div([
                        html_div([
                            html_label("Model name"),
                            dcc_input(id="builder-model-name", type="text", value="custom_growth_model", style=Dict("width" => "100%")),
                        ]),
                        html_div([
                            html_label("Family label"),
                            dcc_dropdown(
                                id="builder-family",
                                options=[Dict("label" => s, "value" => s) for s in ["custom", "logistic", "gompertz", "theta_logistic", "coculture", "mechanistic"]],
                                value="custom", clearable=false,
                                style=Dict("fontSize" => "13px", "color" => "#0f172a", "backgroundColor" => "#ffffff"),
                            ),
                        ]),
                    ]; style=Dict("display" => "grid", "gridTemplateColumns" => "1fr 1fr", "gap" => "16px")),
                    html_br(),
                    html_div([
                        html_div([
                            html_label("State names"),
                            _help("Comma-separated. E.g. `N` for single-state, `S, R` for sensitive/resistant."),
                            dcc_input(id="builder-state-names", type="text", value="N", style=Dict("width" => "100%", "fontFamily" => "monospace")),
                        ]),
                        html_div([
                            html_label("Observable expression"),
                            _help("What is measured. E.g. `N`, or `S + R` for total cells."),
                            dcc_input(id="builder-observable", type="text", value="N", style=Dict("width" => "100%", "fontFamily" => "monospace")),
                        ]),
                    ]; style=Dict("display" => "grid", "gridTemplateColumns" => "1fr 1fr", "gap" => "16px")),
                    html_br(),
                    html_div([
                        html_div([
                            html_label("Parameter names"),
                            dcc_input(id="builder-param-names", type="text", value="r, K", style=Dict("width" => "100%", "fontFamily" => "monospace")),
                        ]),
                        html_div([
                            html_label("Preset constants (name=value)"),
                            _help("Optional fixed values used in equations but not fitted. Example: `hill=1.0, ic50=0.5`."),
                            dcc_input(id="builder-constants", type="text", value="", style=Dict("width" => "100%", "fontFamily" => "monospace")),
                        ]),
                        html_div([
                            html_label("Lower bounds"),
                            dcc_input(id="builder-lower-bounds", type="text", value="1e-6, 1e-3", style=Dict("width" => "100%", "fontFamily" => "monospace")),
                        ]),
                        html_div([
                            html_label("Upper bounds"),
                            dcc_input(id="builder-upper-bounds", type="text", value="5.0, 1e7", style=Dict("width" => "100%", "fontFamily" => "monospace")),
                        ]),
                    ]; style=Dict("display" => "grid", "gridTemplateColumns" => "1fr 1fr", "gap" => "16px")),
                    html_br(),
                    html_label("Math keyboard / block inserter"),
                    _help("These buttons insert snippets at the cursor position inside the equations editor below."),
                    html_div([
                        _keybtn("N", "builder-key-N"), _keybtn("S", "builder-key-S"), _keybtn("R", "builder-key-R"), _keybtn("D", "builder-key-D"), _keybtn("A", "builder-key-A"),
                        _keybtn("E", "builder-key-E"), _keybtn("t", "builder-key-t"), _keybtn("+", "builder-key-plus"), _keybtn("-", "builder-key-minus"), _keybtn("*", "builder-key-times"),
                        _keybtn("/", "builder-key-div"), _keybtn("^", "builder-key-pow"), _keybtn("(", "builder-key-lpar"), _keybtn(")", "builder-key-rpar"),
                        _keybtn("log()", "builder-key-log"), _keybtn("exp()", "builder-key-exp"), _keybtn("growth block", "builder-key-logistic"),
                        _keybtn("Hill block", "builder-key-hill"), _keybtn("competition", "builder-key-competition"), _keybtn("conversion", "builder-key-conversion"),
                    ]),
                    html_label("State equations"),
                    _help("Write one equation per line as `state = expression`. Use parameter names, state names, `t` for time, and `E` for exposure/dose."),
                    dcc_textarea(id="builder-equations", value="N = r*N*(1 - N/K)",
                        style=Dict("width" => "100%", "height" => "150px", "fontFamily" => "monospace", "fontSize" => "14px")),
                    html_br(),
                    _btn("Register built model", "btn-register-built-model"),
                    html_small("✓ Models registered here are saved automatically and restored on GUI restart.",
                        style=Dict("color" => "#0f766e", "display" => "block", "marginTop" => "6px")),
                    html_div(id="builder-preview", children=html_p("Equation preview will appear here after you start typing equations.", style=Dict("color" => "#6b7280"))),
                    html_div(id="custom-model-register-status", children=html_small("No custom model registered yet.", style=Dict("color" => "#6b7280"))),
                ]; title="GUI Model Builder"),

                _card([
                    html_h5("Advanced: Register Custom Models From File", style=Dict("marginTop" => "0")),
                    _help("Optional developer path. Use this only when you already have a Julia file that defines and registers models."),
                    dcc_input(id="custom-model-module-path", type="text", debounce=true,
                        placeholder="/path/to/my_custom_models.jl",
                        style=Dict("width" => "100%", "fontFamily" => "monospace", "fontSize" => "13px")),
                    html_br(),
                    _btn("Register custom models from file", "btn-register-model-module"),
                ]; title="File-Based Registration (Advanced)"),
            ])
        ]),

        # ── Tab 4: Select Models per Stage ───────────────────────────────────
        dcc_tab(label="4. Select Models per Stage", value="tab-models",
            style=_tab_style, selected_style=_tab_selected, children=[
            html_div([
                html_br(),
                _card([
                    html_h5("Choose models for this stage", style=Dict("marginTop" => "0")),
                    _help("Select one or more models for the current stage. In staged workflows, repeat this step per stage."),
                    dcc_dropdown(id="model-select", options=_model_options(), value=_default_models(), multi=true,
                        placeholder="Select models…", style=Dict("fontSize" => "13px")),
                    html_br(),
                    html_h5("Optimizer settings"),
                    _help("More starts and iterations improve fit quality but take longer."),
                    html_div([
                        html_div([
                            html_label("Optimization starts"),
                            dcc_input(id="n-starts", type="number", value=8, min=1, max=200, style=Dict("width" => "100%")),
                        ]),
                        html_div([
                            html_label("Max iterations per start"),
                            dcc_input(id="maxiters", type="number", value=300, min=20, max=10000, style=Dict("width" => "100%")),
                        ]),
                    ]; style=Dict("display" => "grid", "gridTemplateColumns" => "1fr 1fr", "gap" => "16px")),
                ]; title="Model Selection"),

                _card([
                    html_h5("Selected model equations", style=Dict("marginTop" => "0")),
                    html_div(id="model-equations", children=html_p("Select models above to see their equations.", style=Dict("color" => "#6b7280"))),
                ]; title="Model Reference"),
            ])
        ]),

        # ── Tab 5: Fit & Rank per Stage ──────────────────────────────────────
        dcc_tab(label="5. Fit & Rank per Stage", value="tab-rank",
            style=_tab_style, selected_style=_tab_selected, children=[
            html_div([
                html_br(),
                _card([
                    html_h5("Fit current stage", style=Dict("marginTop" => "0")),
                    _help("Fit one stage at a time. After each stage, use the variable mapping panel to map fitted outputs into the next stage model."),
                    dcc_dropdown(id="condition-select", options=[], value=nothing,
                        placeholder="Load data first…", style=Dict("fontSize" => "13px")),
                    html_br(),
                    html_h5("Validation and uncertainty settings", style=Dict("marginTop" => "0")),
                    html_div([
                        html_div([
                            html_label("Train fraction"),
                            dcc_input(id="train-frac", type="number", value=0.7, min=0.5, max=0.95, step=0.05, style=Dict("width" => "100%")),
                        ]),
                        html_div([
                            html_label("Split mode"),
                            dcc_dropdown(id="split-mode",
                                options=[
                                    Dict("label" => "Temporal (early → train, late → validation)", "value" => "temporal"),
                                    Dict("label" => "Random (seeded)", "value" => "random"),
                                ],
                                value="temporal", clearable=false, style=Dict("fontSize" => "13px"),
                            ),
                        ]),
                        html_div([
                            html_label("Bootstrap samples for interval"),
                            dcc_input(id="uncertainty-boot", type="number", value=30, min=5, max=200, step=1, style=Dict("width" => "100%")),
                        ]),
                    ]; style=Dict("display" => "grid", "gridTemplateColumns" => "1fr 1fr 1fr", "gap" => "16px")),
                    html_br(),
                    _btn("Run ranking & fit", "btn-rank"),
                    _btn("Auto-run configured pipeline", "btn-pipeline"),
                ]; title="Stage Fit Runner"),

                _card([
                    html_h5("Inter-stage variable mapping", style=Dict("marginTop" => "0")),
                    _help("After a stage fit completes, map outputs to next-stage variables (e.g., r → r_prey, K → K_prey)."),
                    html_div(id="pipeline-output", children=html_p("Run a stage fit to configure mappings.", style=Dict("color" => "#6b7280"))),
                ]; title="Mapping to Next Stage"),

                dcc_loading(id="rank-spinner", type="circle", color="#0f766e",
                    children=html_div(id="rank-output", children=html_p("Run ranking to see results.", style=Dict("color" => "#6b7280")))),

                html_section([
                    html_h4("Model vs Data"),
                    dcc_graph(id="fit-plot", figure=Dict("data" => Any[], "layout" => Dict("title" => "Run ranking to see fits"))),
                ]),
            ])
        ]),

        # ── Tab 6: Complete Analysis by Stage ────────────────────────────────
        dcc_tab(label="6. Complete Analysis by Stage", value="tab-staged",
            style=_tab_style, selected_style=_tab_selected, children=[
            html_div([
                html_br(),
                _card([
                    html_h5("Stage-organized analysis", style=Dict("marginTop" => "0")),
                    _help("Review all stage results in order, including diagnostics and fitted summaries."),
                    _btn("Refresh staged analysis", "btn-staged"),
                ]; title="Staged Analysis"),
                dcc_loading(id="staged-spinner", type="circle", color="#0f766e",
                    children=html_div(id="staged-output",
                        children=html_p("Run stage fits from Tab 5, then view complete stage-organized analysis here.",
                            style=Dict("color" => "#6b7280")))),
            ])
        ]),

    ]),

]; style=Dict("maxWidth" => "1200px", "margin" => "0 auto", "padding" => "24px 24px 60px 24px"))

# ══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. File loading → csv-path-store ─────────────────────────────────────────
callback!(
    app,
    Output("csv-path-store", "data"),
    Input("upload-data", "contents"),
    Input("btn-ex-basic",       "n_clicks"),
    Input("btn-ex-staged-mono", "n_clicks"),
    Input("btn-ex-staged-co",   "n_clicks"),
    Input("manual-path", "value"),
    State("upload-data", "filename"),
) do contents, n_basic, n_staged_mono, n_staged_co, manual_path, filename
    tid = _triggered_id()
    if tid == "upload-data" && !isnothing(contents)
        fname = isnothing(filename) ? "upload.csv" : filename
        try
            return _save_upload(String(contents), String(fname))
        catch
            return nothing
        end
    elseif tid == "btn-ex-basic"
        return joinpath(EXAMPLE_DIR, "basic_pipeline.csv")
    elseif tid == "btn-ex-staged-mono"
        return joinpath(EXAMPLE_DIR, "staged_monoculture.csv")
    elseif tid == "btn-ex-staged-co"
        return joinpath(EXAMPLE_DIR, "coculture_stages.csv")
    elseif tid == "manual-path" && !isnothing(manual_path) && !isempty(strip(String(manual_path)))
        p = strip(String(manual_path))
        return isfile(p) ? p : nothing
    end
    return nothing
end

# ── 2. csv-path-store → status bar + preflight + condition dropdown ───────────
callback!(
    app,
    Output("status-bar",        "data"),
    Output("load-data-preview", "children"),
    Output("load-tab-output",   "children"),
    Output("conditions-store",  "data"),
    Output("condition-select",  "options"),
    Output("condition-select",  "value"),
    Input("csv-path-store", "data"),
) do path
    no_status   = _alert("No data loaded. Use the Load Data tab to get started.", kind=:warn)
    no_preview  = html_p("Load a dataset above to preview the raw data that will be fitted.",
        style=Dict("color" => "#6b7280"))
    no_preflight = html_p("Load a dataset to see preflight quality checks here.",
        style=Dict("color" => "#6b7280"))

    (isnothing(path) || isempty(String(path))) && return (no_status, no_preview, no_preflight, nothing, [], nothing)
    p = String(path)
    isfile(p) || return (_alert("File not found: $(p)", kind=:error), no_preview, no_preflight, nothing, [], nothing)

    try
        df         = _safe_load(p)
        conditions = build_conditions(df)
        cond_names = [c.name for c in conditions]
        opts       = [Dict("label" => n, "value" => n) for n in cond_names]
        first_val  = isempty(cond_names) ? nothing : cond_names[1]
        staged     = _has_stage_metadata(df)

        status = _alert("Loaded: $(basename(p))  |  $(nrow(df)) rows  |  $(length(conditions)) condition(s)" *
            (staged ? "  |  Staged-pipeline ready ✓" : ""))
        preview = html_div([
            _alert("Data preview ready below. The four graphs show exactly what the fitting routines will see for this dataset."),
            _data_overview_panels(df),
        ])
        return (status, preview, _preflight_output(p), cond_names, opts, first_val)
    catch err
        msg = sprint(showerror, err)
        return (_alert("Error loading file: $(msg)", kind=:error), no_preview, no_preflight, nothing, [], nothing)
    end
end

# ── 2b. Pipeline Designer: stage list and persistence ───────────────────────
callback!(
    app,
    Output("pipeline-stages-store", "data"),
    Output("pipeline-stage-select", "options"),
    Output("pipeline-stage-select", "value"),
    Input("btn-add-stage", "n_clicks"),
    Input("btn-stage-up", "n_clicks"),
    Input("btn-stage-down", "n_clicks"),
    Input("btn-stage-remove", "n_clicks"),
    Input("btn-stage-use-current", "n_clicks"),
    Input("btn-stage-next", "n_clicks"),
    State("pipeline-stages-store", "data"),
    State("pipeline-stage-select", "value"),
    State("csv-path-store", "data"),
    State("model-select", "value"),
) do n_add, n_up, n_down, n_remove, n_use_current, n_next, store_data, selected_idx, csv_path, selected_models
    data = isnothing(store_data) ? Dict("stages" => Any[]) : Dict(store_data)
    stages = haskey(data, "stages") ? Any[data["stages"]...] : Any[]

    function _stage_options(arr)
        [Dict("label" => "$(i). $(get(Dict(s), "name", "Stage $(i)"))", "value" => i) for (i, s) in enumerate(arr)]
    end

    function _default_model()
        if !isnothing(selected_models) && !isempty(selected_models)
            return String(selected_models[1])
        end
        available = list_models()
        return isempty(available) ? "" : String(available[1])
    end

    tid = _triggered_id()
    idx = (isnothing(selected_idx) || Int(selected_idx) < 1) ? 1 : Int(selected_idx)

    if tid == "btn-add-stage"
        push!(stages, Dict(
            "name" => "Stage $(length(stages) + 1)",
            "csv_file" => (isnothing(csv_path) ? "" : String(csv_path)),
            "model_name" => _default_model(),
            "param_mapping" => Dict{String, String}(),
        ))
        idx = length(stages)
    elseif tid == "btn-stage-up" && idx > 1 && idx <= length(stages)
        stages[idx - 1], stages[idx] = stages[idx], stages[idx - 1]
        idx -= 1
    elseif tid == "btn-stage-down" && idx >= 1 && idx < length(stages)
        stages[idx + 1], stages[idx] = stages[idx], stages[idx + 1]
        idx += 1
    elseif tid == "btn-stage-remove" && idx >= 1 && idx <= length(stages)
        deleteat!(stages, idx)
        idx = isempty(stages) ? nothing : min(idx, length(stages))
    elseif tid == "btn-stage-use-current" && idx >= 1 && idx <= length(stages)
        s = Dict(stages[idx])
        s["csv_file"] = isnothing(csv_path) ? "" : String(csv_path)
        s["model_name"] = _default_model()
        stages[idx] = s
    elseif tid == "btn-stage-next" && idx >= 1 && idx < length(stages)
        # Advance to next stage
        idx += 1
    end

    data["stages"] = stages
    opts = _stage_options(stages)
    sel = isempty(stages) ? nothing : (isnothing(idx) ? 1 : idx)
    return data, opts, sel
end

callback!(
    app,
    Output("pipeline-stages-container", "children"),
    Output("pipeline-flowchart", "children"),
    Input("pipeline-stages-store", "data"),
    Input("btn-save-pipeline", "n_clicks"),
    State("pipeline-name", "value"),
) do store_data, save_clicks, pipeline_name
    data = isnothing(store_data) ? Dict("stages" => Any[]) : Dict(store_data)
    stages_raw = haskey(data, "stages") ? Any[data["stages"]...] : Any[]

    stage_cards = Any[]
    flow_nodes = String[]
    for (i, sraw) in enumerate(stages_raw)
        s = Dict(sraw)
        sname = get(s, "name", "Stage $(i)")
        sfile = get(s, "csv_file", "")
        smodel = get(s, "model_name", "")
        push!(stage_cards, _card([
            html_strong(String(sname)),
            html_p("CSV: " * (isempty(String(sfile)) ? "(not set)" : String(sfile)); style=Dict("margin" => "4px 0", "fontSize" => "12px", "color" => "#374151")),
            html_p("Model: " * (isempty(String(smodel)) ? "(not set)" : String(smodel)); style=Dict("margin" => "4px 0", "fontSize" => "12px", "color" => "#374151")),
            html_p("Tip: select stage above, then click 'Use Current Data+Model' after choosing data/model in other tabs."; style=Dict("margin" => "4px 0", "fontSize" => "11px", "color" => "#6b7280")),
        ]; title="Stage $(i)"))
        push!(flow_nodes, String(sname))
    end

    if isempty(stage_cards)
        stage_cards = [html_p("Click 'Add Stage' to begin configuring your pipeline.", style=Dict("color" => "#6b7280"))]
    end

    tid = _triggered_id()
    saved_banner = nothing
    if tid == "btn-save-pipeline" && !isnothing(pipeline_name) && !isempty(strip(String(pipeline_name)))
        pname = strip(String(pipeline_name))
        p_stages = PipelineStage[]
        for (i, sraw) in enumerate(stages_raw)
            s = Dict(sraw)
            sname = String(get(s, "name", "Stage $(i)"))
            sfile = String(get(s, "csv_file", ""))
            smodel = String(get(s, "model_name", ""))
            smap = haskey(s, "param_mapping") ? Dict{String, String}(string(k) => string(v) for (k, v) in pairs(Dict(s["param_mapping"]))) : Dict{String, String}()
            push!(p_stages, PipelineStage(sname, sfile, smodel, smap))
        end
        _save_pipeline(Pipeline(pname, p_stages))
        saved_banner = _alert("Saved pipeline '$(pname)' with $(length(p_stages)) stage(s).")
    end

    flow_text = isempty(flow_nodes) ? "No stages yet." : join(flow_nodes, " → ")
    flow_children = Any[
        html_p(flow_text; style=Dict("color" => "#374151", "fontWeight" => "600", "textAlign" => "center", "padding" => "10px")),
    ]
    if !isnothing(saved_banner)
        push!(flow_children, saved_banner)
    end

    return html_div(stage_cards), html_div(flow_children)
end

# ── 3. Model selection → equations panel ─────────────────────────────────────
callback!(
    app,
    Output("model-equations", "children"),
    Input("model-select", "value"),
) do selected
    (isnothing(selected) || isempty(selected)) &&
        return html_p("Select models to see their equations.", style=Dict("color" => "#6b7280"))
    models = String.(selected)

    items = Any[]
    for m in models
        latex      = _model_latex(m)
        if !(m in list_models())
            continue
        end
        spec = get_model(m)
        family     = spec.base_growth_family
        params     = spec.param_names
        param_str  = join(string.(params), ", ")

        push!(items, html_div([
            html_div([
                html_strong(m),
                html_span("  family: $(family)",
                    style=Dict("color" => "#6b7280", "fontSize" => "11px", "marginLeft" => "8px")),
            ]),
            html_p("Parameters: $(param_str)",
                style=Dict("fontSize" => "12px", "color" => "#374151", "margin" => "2px 0 4px 0")),
            dcc_markdown(latex, mathjax=true),
        ]; style=Dict(
            "borderLeft" => "3px solid #0f766e",
            "marginBottom" => "14px",
            "background"   => "#f9fafb",
            "padding"      => "10px 14px",
            "borderRadius" => "0 6px 6px 0",
        )))
    end
    return html_div(items)
end

callback!(
    app,
    Output("builder-family", "value"),
    Output("builder-state-names", "value"),
    Output("builder-observable", "value"),
    Output("builder-param-names", "value"),
    Output("builder-constants", "value"),
    Output("builder-lower-bounds", "value"),
    Output("builder-upper-bounds", "value"),
    Output("builder-equations", "value"),
    Input("btn-load-builder-template", "n_clicks"),
    State("builder-template", "value"),
    State("builder-family", "value"),
    State("builder-state-names", "value"),
    State("builder-observable", "value"),
    State("builder-param-names", "value"),
    State("builder-constants", "value"),
    State("builder-lower-bounds", "value"),
    State("builder-upper-bounds", "value"),
    State("builder-equations", "value"),
) do n_clicks, template, cur_family, cur_states, cur_observable, cur_params, cur_constants, cur_lb, cur_ub, cur_equations
    n_clicks == 0 && return (cur_family, cur_states, cur_observable, cur_params, cur_constants, cur_lb, cur_ub, cur_equations)
    tpl = _builder_template_data(isnothing(template) ? "logistic" : String(template))
    return (tpl.family, tpl.states, tpl.observable, tpl.params, tpl.constants, tpl.lower, tpl.upper, tpl.equations)
end

callback!(
    app,
    Output("builder-preview", "children"),
    Input("builder-model-name", "value"),
    Input("builder-state-names", "value"),
    Input("builder-observable", "value"),
    Input("builder-param-names", "value"),
    Input("builder-constants", "value"),
    Input("builder-equations", "value"),
) do name, state_text, observable_text, params_text, constants_text, equations_text
    try
        return _custom_model_preview(name, state_text, observable_text, params_text, constants_text, equations_text)
    catch err
        return _alert("Preview unavailable: $(sprint(showerror, err))", kind=:warn)
    end
end

# ── 4. Custom model registration → refresh model dropdown ─────────────────────
# TEMPORARILY DISABLED FOR DEBUGGING - callback removed
# This callback was causing MethodError with 12 parameters being called with 3

# ── Pipeline mapping helpers ─────────────────────────────────────────────────
function _param_names_for_model(model_name::AbstractString)
    m = String(model_name)
    if !(m in list_models())
        return String[]
    end
    return String.(get_model(m).param_names)
end

function _suggest_mapping_rows(source_model::AbstractString, target_model::AbstractString)
    src = _param_names_for_model(source_model)
    dst = _param_names_for_model(target_model)
    rows = Any[]
    for s in src
        match_exact = findfirst(d -> lowercase(d) == lowercase(s), dst)
        match_partial = isnothing(match_exact) ? findfirst(d -> occursin(lowercase(s), lowercase(d)) || occursin(lowercase(d), lowercase(s)), dst) : nothing
        target = isnothing(match_exact) ? (isnothing(match_partial) ? "(unmapped)" : dst[match_partial]) : dst[match_exact]
        push!(rows, html_tr([html_td(s), html_td("→"), html_td(target)]))
    end
    isempty(rows) && push!(rows, html_tr([html_td("No parameters found"), html_td(""), html_td("")]))
    return rows
end

function _mapping_prompt_panel(stage_name::AbstractString, source_model::AbstractString, next_stage_name::AbstractString, target_model::AbstractString)
    return _card([
        html_h5("Stage complete: $(stage_name)", style=Dict("marginTop" => "0")),
        html_p("Map fitted variables from $(source_model) into next stage $(next_stage_name) model $(target_model)."; style=Dict("margin" => "4px 0 10px 0")),
        html_table([
            html_thead(html_tr([html_th("From current stage"), html_th(""), html_th("To next stage")])) ,
            html_tbody(_suggest_mapping_rows(source_model, target_model)),
        ]; style=Dict("width" => "100%", "fontSize" => "12px")),
        html_small("Use this mapping table as the prompt to decide parameter transfer before running the next stage."; style=Dict("color" => "#6b7280")),
    ]; title="Inter-stage mapping prompt")
end

function _pipeline_autorun_panel(path::AbstractString, stages_raw, n_starts::Int, maxiters::Int, train_frac::Float64, split_mode::String, uncertainty_boot::Int)
    cards = Any[]
    for (i, sraw) in enumerate(stages_raw)
        s = Dict(sraw)
        sname = String(get(s, "name", "Stage $(i)"))
        scsv = String(get(s, "csv_file", path))
        smodel = String(get(s, "model_name", ""))
        models = isempty(smodel) ? _default_models() : [smodel]
        if isnothing(scsv) || isempty(scsv) || !isfile(scsv)
            push!(cards, _alert("$(sname): CSV not found. Skipped.", kind=:warn))
            continue
        end
        try
            conds = build_conditions(_safe_load(scsv))
            cond_name = isempty(conds) ? "" : String(conds[1].name)
            panel, _ = _rank_output(scsv, models, cond_name, n_starts, maxiters, train_frac, split_mode, uncertainty_boot)
            push!(cards, _card([
                html_h5("$(sname) ✓", style=Dict("marginTop" => "0")),
                html_p("Model: $(join(models, ", ")) | Condition: $(isempty(cond_name) ? "(none)" : cond_name)"; style=Dict("fontSize" => "12px", "color" => "#374151")),
                panel,
            ]; title="Auto-run result"))
        catch err
            push!(cards, _alert("$(sname): auto-run failed: $(sprint(showerror, err))", kind=:error))
        end
        if i < length(stages_raw)
            next_s = Dict(stages_raw[i + 1])
            next_name = String(get(next_s, "name", "Stage $(i+1)"))
            next_model = String(get(next_s, "model_name", ""))
            push!(cards, _mapping_prompt_panel(sname, isempty(smodel) ? (isempty(models) ? "" : models[1]) : smodel, next_name, next_model))
        end
    end
    isempty(cards) && return html_p("No configured stages to auto-run.", style=Dict("color" => "#6b7280"))
    return html_div(cards)
end

# ── 5. Rank & Fit ─────────────────────────────────────────────────────────────
callback!(
    app,
    Output("rank-output", "children"),
    Output("fit-plot",    "figure"),
    Input("btn-rank", "n_clicks"),
    State("csv-path-store",    "data"),
    State("model-select",      "value"),
    State("condition-select",  "value"),
    State("n-starts",          "value"),
    State("maxiters",          "value"),
    State("train-frac",        "value"),
    State("split-mode",        "value"),
    State("uncertainty-boot",  "value"),
) do n_clicks, path, models, cond_name, n_starts, maxiters, train_frac, split_mode, uncertainty_boot
    base_fig = Dict("data" => Any[], "layout" => Dict("title" => "Run ranking to see fits"))
    n_clicks == 0 && return (html_p("Click 'Run ranking & fit' to see results.", style=Dict("color" => "#6b7280")), base_fig)
    (isnothing(path) || !isfile(String(path))) && return (_alert("Load a dataset first (Tab 1).", kind=:warn), base_fig)
    try
        ms  = isnothing(models)    || isempty(models)    ? _default_models() : String.(models)
        cn  = isnothing(cond_name) ? "" : String(cond_name)
        ns  = isnothing(n_starts)  ? 8   : Int(n_starts)
        mi  = isnothing(maxiters)  ? 300 : Int(maxiters)
        tf  = isnothing(train_frac) ? 0.7 : Float64(train_frac)
        sm  = isnothing(split_mode) ? "temporal" : String(split_mode)
        nb  = isnothing(uncertainty_boot) ? 30 : Int(uncertainty_boot)
        panel, fig = _rank_output(String(path), ms, cn, ns, mi, tf, sm, nb)
        return (panel, fig)
    catch err
        msg = sprint(showerror, err)
        return (_alert("Ranking failed: $(msg)\n\nTip: Run preflight checks first (Tab 1).", kind=:error), base_fig)
    end
end

# ── 5b. Mapping prompt + auto-run staged flow ────────────────────────────────
callback!(
    app,
    Output("pipeline-output", "children"),
    Input("btn-rank", "n_clicks"),
    Input("btn-pipeline", "n_clicks"),
    State("csv-path-store", "data"),
    State("model-select", "value"),
    State("pipeline-stages-store", "data"),
    State("pipeline-stage-select", "value"),
    State("n-starts", "value"),
    State("maxiters", "value"),
    State("train-frac", "value"),
    State("split-mode", "value"),
    State("uncertainty-boot", "value"),
) do n_rank, n_pipeline, path, models, stages_data, selected_stage_idx, n_starts, maxiters, train_frac, split_mode, uncertainty_boot
    n_rank == 0 && n_pipeline == 0 && return html_p("Run a stage fit to configure mappings.", style=Dict("color" => "#6b7280"))

    if isnothing(stages_data) || !haskey(Dict(stages_data), "stages")
        return _alert("No pipeline stages configured. Use Tab 2 to add stages first.", kind=:warn)
    end

    stages_raw = Any[Dict(stages_data)["stages"]...]
    isempty(stages_raw) && return _alert("No pipeline stages configured. Use Tab 2 to add stages first.", kind=:warn)

    tid = _triggered_id()
    if tid == "btn-pipeline"
        (isnothing(path) || !isfile(String(path))) && return _alert("Load a dataset first (Tab 1).", kind=:warn)
        ns = isnothing(n_starts) ? 8 : Int(n_starts)
        mi = isnothing(maxiters) ? 300 : Int(maxiters)
        tf = isnothing(train_frac) ? 0.7 : Float64(train_frac)
        sm = isnothing(split_mode) ? "temporal" : String(split_mode)
        nb = isnothing(uncertainty_boot) ? 30 : Int(uncertainty_boot)
        return _pipeline_autorun_panel(String(path), stages_raw, ns, mi, tf, sm, nb)
    end

    idx = (isnothing(selected_stage_idx) ? 1 : Int(selected_stage_idx))
    idx = clamp(idx, 1, length(stages_raw))
    if idx == length(stages_raw)
        s = Dict(stages_raw[idx])
        sname = String(get(s, "name", "Stage $(idx)"))
        return _alert("$(sname) is the final stage. No next-stage mapping required.")
    end

    current = Dict(stages_raw[idx])
    nxt = Dict(stages_raw[idx + 1])
    source_model = if !isnothing(models) && !isempty(models)
        String(models[1])
    else
        String(get(current, "model_name", ""))
    end
    target_model = String(get(nxt, "model_name", ""))
    stage_name = String(get(current, "name", "Stage $(idx)"))
    next_name = String(get(nxt, "name", "Stage $(idx + 1)"))
    return _mapping_prompt_panel(stage_name, source_model, next_name, target_model)
end

# ── 6. Staged Pipeline ────────────────────────────────────────────────────────
callback!(
    app,
    Output("staged-output", "children"),
    Input("btn-staged", "n_clicks"),
    State("csv-path-store", "data"),
) do n_clicks, path
    n_clicks == 0 && return html_p("Click 'Run Staged Pipeline' to start.", style=Dict("color" => "#6b7280"))
    (isnothing(path) || !isfile(String(path))) && return _alert("Load a dataset first (Tab 1).", kind=:warn)
    try
        return _staged_output(String(path))
    catch err
        return _alert("Staged pipeline failed: $(sprint(showerror, err))", kind=:error)
    end
end

# ── Glossary toggle ───────────────────────────────────────────────────────────
callback!(
    app,
    Output("glossary-panel", "style"),
    Input("glossary-toggle", "n_clicks"),
    State("glossary-panel", "style"),
) do n_clicks, current_style
    hidden  = Dict("display" => "none", "border" => "1px solid #d1fae5", "borderRadius" => "8px",
                   "padding" => "12px 18px", "marginTop" => "8px", "background" => "#f0fdf4",
                   "boxShadow" => "0 2px 8px rgba(0,0,0,0.07)")
    visible = merge(hidden, Dict("display" => "block"))
    n_clicks == 0 && return hidden
    current_display = get(current_style, "display", "none")
    return current_display == "none" ? visible : hidden
end

# ══════════════════════════════════════════════════════════════════════════════
# Launch
# ══════════════════════════════════════════════════════════════════════════════

if abspath(PROGRAM_FILE) == @__FILE__
    println("GrowthParameterEstimation GUI running at http://$(HOST):$(PORT)")
    try
        run_server(app, HOST, PORT; debug=false)
    catch err
        msg = sprint(showerror, err)
        if occursin("EADDRINUSE", msg)
            println("Port $(PORT) is already in use. Set GPE_GUI_PORT to another port and restart.")
        end
        rethrow(err)
    end
end

