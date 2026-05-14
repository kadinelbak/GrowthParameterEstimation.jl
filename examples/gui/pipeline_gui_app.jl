using Dash
using CSV
using DataFrames
using GrowthParameterEstimation
using Statistics
using Base64
using TOML
using Random

const HOST = "127.0.0.1"
const PORT = parse(Int, get(ENV, "GPE_GUI_PORT", "8050"))
const LATEX_CONFIG_PATH = joinpath(dirname(dirname(dirname(@__FILE__))), "config", "model_latex.toml")
const EXAMPLE_DIR = joinpath(dirname(@__FILE__), "data")

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

function _glossary()
    html_details([
        html_summary("📖  Glossary — click to expand",
            style=Dict("cursor" => "pointer", "fontWeight" => "600", "color" => "#0f766e")),
        html_div([
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
        ]; style=Dict("marginTop" => "12px", "padding" => "0 12px")),
    ]; style=Dict("border" => "1px solid #d1fae5", "padding" => "10px 14px", "borderRadius" => "6px", "marginBottom" => "20px"))
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

    # ── Header ────────────────────────────────────────────────────────────────
    html_header([
        html_h2("GrowthParameterEstimation",
            style=Dict("margin" => "0", "color" => "#0f766e")),
        html_p("Fit, compare, and rank ODE growth models against your cell-count time series.",
            style=Dict("margin" => "4px 0 0 0", "color" => "#6b7280")),
    ]; style=Dict("borderBottom" => "2px solid #d1fae5", "marginBottom" => "20px", "paddingBottom" => "12px")),

    # ── Status bar ────────────────────────────────────────────────────────────
    html_div(id="status-bar",
        children=_alert("No data loaded. Use the Load Data tab to get started.", kind=:warn)),

    # ── Glossary (collapsible, always accessible) ─────────────────────────────
    _glossary(),

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

        # ── Tab 2: Select Models ──────────────────────────────────────────────
        dcc_tab(label="2. Select Models", value="tab-models",
            style=_tab_style, selected_style=_tab_selected, children=[
            html_div([
                html_br(),
                _card([
                    html_h5("Choose models to fit", style=Dict("marginTop" => "0")),
                    _help("Select one or more models. Models are grouped by mathematical family. Selecting from the same family compares structural variants; selecting across families compares modelling assumptions."),
                    dcc_dropdown(
                        id="model-select",
                        options=_model_options(),
                        value=_default_models(),
                        multi=true,
                        placeholder="Select models…",
                        style=Dict("fontSize" => "13px")),
                    html_br(),
                    html_h5("Register custom models from file"),
                    _help("Point to a Julia file that either calls register_model! directly or defines register_custom_models!()."),
                    dcc_input(id="custom-model-module-path", type="text", debounce=true,
                        placeholder="/path/to/my_custom_models.jl",
                        style=Dict("width" => "100%", "fontFamily" => "monospace", "fontSize" => "13px")),
                    html_br(),
                    _btn("Register custom models", "btn-register-model-module"),
                    html_div(id="custom-model-register-status",
                        children=html_small("No custom model file loaded yet.", style=Dict("color" => "#6b7280"))),
                    html_br(),
                    html_h5("Optimizer settings"),
                    _help("More starts and iterations improve fit quality but take longer. Defaults (8 starts, 300 iterations) are good for quick exploration."),
                    html_div([
                        html_div([
                            html_label("Optimization starts"),
                            dcc_input(id="n-starts", type="number", value=8, min=1, max=200,
                                style=Dict("width" => "100%")),
                        ]),
                        html_div([
                            html_label("Max iterations per start"),
                            dcc_input(id="maxiters", type="number", value=300, min=20, max=10000,
                                style=Dict("width" => "100%")),
                        ]),
                    ]; style=Dict("display" => "grid", "gridTemplateColumns" => "1fr 1fr", "gap" => "16px")),
                ]; title="Model Selection"),

                _card([
                    html_h5("Selected model equations", style=Dict("marginTop" => "0")),
                    _help("ODE equations and parameter names for each selected model. Equations are rendered automatically."),
                    html_div(id="model-equations",
                        children=html_p("Select models above to see their equations.",
                            style=Dict("color" => "#6b7280"))),
                ]; title="Model Reference"),
            ])
        ]),

        # ── Tab 3: Fit & Rank ─────────────────────────────────────────────────
        dcc_tab(label="3. Fit & Rank", value="tab-rank",
            style=_tab_style, selected_style=_tab_selected, children=[
            html_div([
                html_br(),
                _card([
                    html_h5("Select a condition and run the ranking", style=Dict("marginTop" => "0")),
                    _help("Each condition is a unique combination of dose, cell line, and replicate. Load data first (Tab 1) to populate this list."),
                    dcc_dropdown(id="condition-select", options=[], value=nothing,
                        placeholder="Load data first…", style=Dict("fontSize" => "13px")),
                    html_br(),
                    html_h5("Validation and uncertainty settings", style=Dict("marginTop" => "0")),
                    html_div([
                        html_div([
                            html_label("Train fraction"),
                            dcc_input(id="train-frac", type="number", value=0.7, min=0.5, max=0.95, step=0.05,
                                style=Dict("width" => "100%")),
                        ]),
                        html_div([
                            html_label("Split mode"),
                            dcc_dropdown(
                                id="split-mode",
                                options=[
                                    Dict("label" => "Temporal (early → train, late → validation)", "value" => "temporal"),
                                    Dict("label" => "Random (seeded)", "value" => "random"),
                                ],
                                value="temporal",
                                clearable=false,
                                style=Dict("fontSize" => "13px"),
                            ),
                        ]),
                        html_div([
                            html_label("Bootstrap samples for interval"),
                            dcc_input(id="uncertainty-boot", type="number", value=30, min=5, max=200, step=1,
                                style=Dict("width" => "100%")),
                        ]),
                    ]; style=Dict("display" => "grid", "gridTemplateColumns" => "1fr 1fr 1fr", "gap" => "16px")),
                    html_br(),
                    _btn("Run ranking & fit", "btn-rank"),
                    _help("Runtime scales with: number of models × number of conditions × starts × max iterations."),
                ]; title="Fit & Rank Models"),

                dcc_loading(id="rank-spinner", type="circle", color="#0f766e",
                    children=html_div(id="rank-output",
                        children=html_p("Run ranking to see results.", style=Dict("color" => "#6b7280")))),

                html_section([
                    html_h4("Model vs Data"),
                    dcc_graph(id="fit-plot",
                        figure=Dict("data" => Any[], "layout" => Dict("title" => "Run ranking to see fits"))),
                ]),
            ])
        ]),

        # ── Tab 4: Full Pipeline ──────────────────────────────────────────────
        dcc_tab(label="4. Full Pipeline", value="tab-pipeline",
            style=_tab_style, selected_style=_tab_selected, children=[
            html_div([
                html_br(),
                _card([
                    html_h5("Run the full end-to-end pipeline", style=Dict("marginTop" => "0")),
                    _help("Runs preflight checks, QC, and model fitting across all conditions for all selected models, then ranks results and exports them to disk."),
                    _btn("Run Full Pipeline", "btn-pipeline"),
                    _help("Results are also saved to results/gui_pipeline/ on disk."),
                ]; title="Full Pipeline"),
                dcc_loading(id="pipeline-spinner", type="circle", color="#0f766e",
                    children=html_div(id="pipeline-output",
                        children=html_p("Click the button above to run the full pipeline.",
                            style=Dict("color" => "#6b7280")))),
            ])
        ]),

        # ── Tab 5: Staged Pipeline ────────────────────────────────────────────
        dcc_tab(label="5. Staged Pipeline", value="tab-staged",
            style=_tab_style, selected_style=_tab_selected, children=[
            html_div([
                html_br(),
                _card([
                    html_h5("Multi-stage workflow", style=Dict("marginTop" => "0")),
                    _help("Runs separate fitting stages for each combination of culture type and population type (e.g. monoculture-naive, co-culture-mixed). Requires culture_type and population_type columns in your CSV. Load one of the 'Staged' example datasets from Tab 1 to try this."),
                    _btn("Run Staged Pipeline", "btn-staged"),
                    _help("Results are saved to results/gui_staged/ on disk."),
                ]; title="Staged Pipeline"),
                dcc_loading(id="staged-spinner", type="circle", color="#0f766e",
                    children=html_div(id="staged-output",
                        children=html_p("Load a staged dataset (Tab 1 → Staged examples) and click the button above.",
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
    Output("status-bar",        "children"),
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

# ── 4. Custom model registration → refresh model dropdown ─────────────────────
callback!(
    app,
    Output("model-select", "options"),
    Output("model-select", "value"),
    Output("custom-model-register-status", "children"),
    Input("btn-register-model-module", "n_clicks"),
    State("custom-model-module-path", "value"),
    State("model-select", "value"),
) do n_clicks, model_path, current_selection
    available = list_models()
    options = _model_options(available)
    selected = isnothing(current_selection) ? String[] : [String(m) for m in current_selection if String(m) in available]
    isempty(selected) && (selected = _default_models())

    if n_clicks == 0
        return options, selected, html_small("No custom model file loaded yet.", style=Dict("color" => "#6b7280"))
    end

    if isnothing(model_path) || isempty(strip(String(model_path)))
        return options, selected, _alert("Provide a valid path to a Julia model file first.", kind=:warn)
    end

    p = strip(String(model_path))
    if !isfile(p)
        return options, selected, _alert("Model file not found: $(p)", kind=:error)
    end

    try
        before = Set(list_models())
        register_models_from_file!(p)
        after = list_models()
        new_models = sort(collect(setdiff(Set(after), before)))

        options = _model_options(after)
        merged = unique(vcat(selected, new_models))
        merged = [m for m in merged if m in after]
        isempty(merged) && (merged = _default_models())

        msg = isempty(new_models) ?
            "Model file loaded. No new names detected (file may overwrite existing models)." :
            "Registered $(length(new_models)) model(s): $(join(new_models, ", "))"
        return options, merged, _alert(msg)
    catch err
        return options, selected, _alert("Custom model registration failed: $(sprint(showerror, err))", kind=:error)
    end
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

# ── 5. Full Pipeline ──────────────────────────────────────────────────────────
callback!(
    app,
    Output("pipeline-output", "children"),
    Input("btn-pipeline", "n_clicks"),
    State("csv-path-store", "data"),
    State("model-select",   "value"),
) do n_clicks, path, models
    n_clicks == 0 && return html_p("Click 'Run Full Pipeline' to start.", style=Dict("color" => "#6b7280"))
    (isnothing(path) || !isfile(String(path))) && return _alert("Load a dataset first (Tab 1).", kind=:warn)
    try
        ms = isnothing(models) || isempty(models) ? DEFAULT_MODELS : String.(models)
        return _pipeline_output(String(path), ms)
    catch err
        return _alert("Pipeline failed: $(sprint(showerror, err))", kind=:error)
    end
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

