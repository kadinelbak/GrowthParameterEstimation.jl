using Dash
using CSV
using DataFrames
using GrowthParameterEstimation
using Statistics

const HOST = "127.0.0.1"
const PORT = parse(Int, get(ENV, "GPE_GUI_PORT", "8050"))
const LATEX_CONFIG_PATH = joinpath(dirname(dirname(dirname(@__FILE__))), "config", "model_latex.toml")

# Cache for LaTeX config (loaded once)
const LATEX_CACHE = Dict{String, String}()
let
    if isfile(LATEX_CONFIG_PATH)
        try
            using TOML
            config = TOML.parsefile(LATEX_CONFIG_PATH)
            if haskey(config, "latex")
                for (model, latex) in config["latex"]
                    LATEX_CACHE[model] = latex
                end
            end
        catch e
            # Silently continue if TOML loading fails
        end
    end
end

function _parse_model_list(txt::AbstractString)
    raw = split(txt, ',')
    models = String[]
    for item in raw
        model = strip(item)
        isempty(model) && continue
        push!(models, model)
    end
    return models
end

function _safe_load(path::AbstractString)
    raw = CSV.read(path, DataFrame)
    return normalize_schema(raw)
end

function _has_stage_metadata(df::DataFrame)
    cols = Set(Symbol.(names(df)))
    return (:culture_type in cols) && (:population_type in cols)
end

function _timing_guidance()
    return html_article([
        html_h4("Runtime Guide"),
        html_ul([
            html_li("Step 2 usually finishes in under a second for the example CSVs. Very large files may take a few seconds because this step mainly validates, groups, and sorts the data."),
            html_li("Step 3 is the first expensive step. Its runtime grows with the number of models, grouped conditions, optimization starts, and max iterations."),
            html_li("With the default GUI settings and a small example dataset, Step 3 is often in the few-seconds to tens-of-seconds range. Larger datasets or more aggressive settings can push it into minutes."),
            html_li("While a step is running, the spinner below will stay visible so users can tell the app is still working."),
        ]),
    ])
end

function _triggered_step_id()
    ctx = callback_context()
    isempty(ctx.triggered) && return nothing

    prop_id = ctx.triggered[1].prop_id
    return first(split(String(prop_id), "."))
end

function _as_table(df::DataFrame; limit::Int = 10)
    shown = nrow(df) > limit ? first(df, limit) : df
    header = html_tr([html_th(string(c)) for c in names(shown)])
    rows = [
        html_tr([html_td(string(shown[i, c])) for c in names(shown)]) for i in 1:nrow(shown)
    ]
    return html_div([
        html_table([
            html_thead(header),
            html_tbody(rows),
        ]; style=Dict("width" => "100%", "borderCollapse" => "collapse")),
        html_small("Showing $(nrow(shown)) of $(nrow(df)) rows"),
    ])
end

function _condition_lookup(conditions::Vector{FitCondition})
    return Dict(c.name => c for c in conditions)
end

function _condition_match_key(name::AbstractString)
    return lowercase(strip(String(name)))
end

function _find_condition_index(conditions::Vector{FitCondition}, name::AbstractString)
    key = _condition_match_key(name)
    return findfirst(c -> _condition_match_key(c.name) == key, conditions)
end

function _read_csv_or_empty(path::AbstractString)
    if isfile(path)
        return CSV.read(path, DataFrame)
    end
    return DataFrame(message=["Missing file: $(path)"])
end

function _residual_diagnostics_df(cond::FitCondition, pred::Vector{Float64}; outlier_threshold::Float64 = 2.0)
    residual = cond.count .- pred
    resid_std = std(residual)
    std_resid = resid_std > 0 ? residual ./ resid_std : fill(0.0, length(residual))

    return DataFrame(
        time=cond.time,
        observed=cond.count,
        predicted=pred,
        residual=residual,
        standardized_residual=std_resid,
        outlier=abs.(std_resid) .> outlier_threshold,
    )
end

function _sensitivity_df(spec::ModelSpec, cond::FitCondition, params::Vector{Float64}; perturbation::Float64 = 0.1)
    baseline = simulate(
        spec,
        cond.time,
        params;
        u0=cond.u0,
        exposure=cond.exposure,
    )

    if !baseline.success
        return DataFrame(
            parameter=String[],
            value=Float64[],
            max_rel_change=Float64[],
            mean_rel_change=Float64[],
            sensitivity_index=Float64[],
            status=String[],
        )
    end

    base_obs = baseline.observed
    out = DataFrame(
        parameter=String[],
        value=Float64[],
        max_rel_change=Float64[],
        mean_rel_change=Float64[],
        sensitivity_index=Float64[],
        status=String[],
    )

    for i in eachindex(params)
        p_up = copy(params)
        p_down = copy(params)

        delta = max(abs(params[i]) * perturbation, 1e-8)
        p_up[i] = params[i] + delta
        p_down[i] = max(1e-12, params[i] - delta)

        up = simulate(spec, cond.time, p_up; u0=cond.u0, exposure=cond.exposure)
        down = simulate(spec, cond.time, p_down; u0=cond.u0, exposure=cond.exposure)

        pname = i <= length(spec.param_names) ? String(spec.param_names[i]) : "p$(i)"

        if up.success && down.success
            rel_up = abs.(up.observed .- base_obs) ./ (abs.(base_obs) .+ 1e-10)
            rel_down = abs.(down.observed .- base_obs) ./ (abs.(base_obs) .+ 1e-10)
            max_rel = max(maximum(rel_up), maximum(rel_down))
            mean_rel = mean(vcat(rel_up, rel_down))
            push!(out, (
                pname,
                params[i],
                max_rel,
                mean_rel,
                max_rel / perturbation,
                "ok",
            ))
        else
            push!(out, (
                pname,
                params[i],
                NaN,
                NaN,
                NaN,
                "solver_failed",
            ))
        end
    end

    sort!(out, :sensitivity_index; rev=true)
    return out
end

function _analysis_panel(ranked, conditions::Vector{FitCondition}, selected_condition::String, selected_model::String)
    c_lookup = _condition_lookup(conditions)
    cond = get(c_lookup, selected_condition, nothing)
    if isnothing(cond)
        cond_idx = _find_condition_index(conditions, selected_condition)
        if !isnothing(cond_idx)
            cond = conditions[cond_idx]
        end
    end
    isnothing(cond) && return html_div([html_h4("Diagnostics & Sensitivity"), html_p("Condition not found in grouped conditions.")])

    haskey(ranked.fits, selected_model) || return html_div([html_h4("Diagnostics & Sensitivity"), html_p("Selected model has no successful fit object.")])
    fit_info = ranked.fits[selected_model]
    hit = findfirst(pc -> _condition_match_key(pc.condition) == _condition_match_key(cond.name) && pc.success, fit_info.per_condition)
    isnothing(hit) && return html_div([html_h4("Diagnostics & Sensitivity"), html_p("No per-condition fit details were found for this condition.")])

    pc = fit_info.per_condition[hit]
    pred = Float64.(pc.observed)
    resid_df = _residual_diagnostics_df(cond, pred)

    rmse = sqrt(mean((resid_df.residual) .^ 2))
    mae = mean(abs.(resid_df.residual))
    mean_resid = mean(resid_df.residual)
    max_abs_resid = maximum(abs.(resid_df.residual))
    n_outliers = sum(resid_df.outlier)

    spec = get_model(selected_model)
    sens_df = _sensitivity_df(spec, cond, Float64.(pc.params))

    sens_fig = if nrow(sens_df) == 0
        Dict(
            "data" => Any[],
            "layout" => Dict("title" => "Sensitivity unavailable for selected model/condition"),
        )
    else
        valid = filter(row -> !isnan(row.sensitivity_index), sens_df)
        Dict(
            "data" => Any[
                Dict(
                    "x" => valid.parameter,
                    "y" => valid.sensitivity_index,
                    "type" => "bar",
                    "name" => "Sensitivity Index",
                ),
            ],
            "layout" => Dict(
                "title" => "Parameter Sensitivity Index (Condition: $(selected_condition))",
                "xaxis" => Dict("title" => "Parameter"),
                "yaxis" => Dict("title" => "Sensitivity Index"),
            ),
        )
    end

    return html_div([
        html_h4("Diagnostics & Sensitivity"),
        html_p("Model: $(selected_model) | Condition: $(selected_condition)"),
        html_p("Residual summary: RMSE=$(round(rmse, digits=4)), MAE=$(round(mae, digits=4)), mean residual=$(round(mean_resid, digits=4)), max |residual|=$(round(max_abs_resid, digits=4)), outliers=$(n_outliers)"),
        html_h5("Residual Diagnostics Table"),
        _as_table(resid_df; limit=50),
        html_h5("Sensitivity Table"),
        _as_table(sens_df; limit=50),
        html_h5("Sensitivity Plot"),
        dcc_graph(id="sensitivity-plot", figure=sens_fig),
    ])
end

function _pipeline_figure_and_key(run)
    overlay_paths = [p for p in run.plots if endswith(p, "_overlay.csv")]
    if isempty(overlay_paths)
        fig = Dict(
            "data" => Any[],
            "layout" => Dict("title" => "No overlay data available from pipeline exports"),
        )
        key_df = DataFrame(series=String[], meaning=String[])
        return fig, key_df
    end

    overlay_df = CSV.read(first(overlay_paths), DataFrame)
    if !(:time in Symbol.(names(overlay_df))) || !(:observed in Symbol.(names(overlay_df)))
        fig = Dict(
            "data" => Any[],
            "layout" => Dict("title" => "Overlay CSV is missing required columns"),
        )
        key_df = DataFrame(series=["observed"], meaning=["Observed data points from input CSV"])
        return fig, key_df
    end

    traces = Any[
        Dict(
            "x" => overlay_df.time,
            "y" => overlay_df.observed,
            "mode" => "markers",
            "name" => "Observed",
        ),
    ]

    key_rows = NamedTuple[(series="Observed", meaning="Observed data points from input CSV")]
    pred_cols = [c for c in names(overlay_df) if startswith(String(c), "pred_")]
    for c in pred_cols
        model_name = replace(String(c), "pred_" => "")
        push!(traces, Dict(
            "x" => overlay_df.time,
            "y" => overlay_df[!, c],
            "mode" => "lines",
            "name" => "Predicted: $(model_name)",
        ))
        push!(key_rows, (series="Predicted: $(model_name)", meaning="Model trajectory from pipeline export"))
    end

    fig = Dict(
        "data" => traces,
        "layout" => Dict(
            "title" => "Pipeline Export Overlay (first condition)",
            "xaxis" => Dict("title" => "Time"),
            "yaxis" => Dict("title" => "Count"),
            "legend" => Dict("orientation" => "h", "y" => -0.2),
        ),
    )

    return fig, DataFrame(key_rows)
end

function _condition_sse_from_pc(pc)
    residuals = Float64.(pc.residuals)
    if isempty(residuals) || any(!isfinite, residuals)
        return Inf
    end
    return sum(residuals .^ 2)
end

function _staged_condition_best_models(run)
    rows = NamedTuple[]

    for stage in run.stages
        if stage.status != "completed" || isnothing(stage.result)
            continue
        end

        for (model_name, fit_info) in pairs(stage.result.fits)
            for pc in fit_info.per_condition
                pc.success || continue
                sse = _condition_sse_from_pc(pc)
                isfinite(sse) || continue
                push!(rows, (
                    stage=String(stage.name),
                    condition=String(pc.condition),
                    model=String(model_name),
                    sse=sse,
                ))
            end
        end
    end

    if isempty(rows)
        return DataFrame(stage=String[], condition=String[], best_model=String[], sse=Float64[])
    end

    raw = DataFrame(rows)
    grouped = groupby(raw, [:stage, :condition])

    best = combine(grouped) do g
        idx = argmin(g.sse)
        (best_model=g.model[idx], sse=g.sse[idx])
    end

    sort!(best, [:stage, :condition])
    return best
end

function _staged_stage_ranking_table(run)
    rows = NamedTuple[]

    for stage in run.stages
        if stage.status != "completed" || isnothing(stage.result)
            continue
        end

        ranking = stage.result.ranking
        for r in eachrow(ranking)
            push!(rows, (
                stage=String(stage.name),
                model=String(r.model),
                bic=Float64(r.bic),
                delta_bic=Float64(r.delta_bic),
                aic=Float64(r.aic),
                sse=Float64(r.sse),
            ))
        end
    end

    if isempty(rows)
        return DataFrame(stage=String[], model=String[], bic=Float64[], delta_bic=Float64[], aic=Float64[], sse=Float64[])
    end

    out = DataFrame(rows)
    sort!(out, [:stage, :bic])
    return out
end

function _model_latex(model_name::AbstractString)
    # Check config file cache first (optional LaTeX)
    if haskey(LATEX_CACHE, String(model_name))
        return LATEX_CACHE[String(model_name)]
    end
    # Fallback: just return the model name if not in config
    return model_name
end

function _staged_figure_and_key(run)
    # Collect overlays from ALL completed stages
    all_overlays = Vector{Tuple{String, String}}()  # (stage_name, path)
    
    for stage in run.stages
        if stage.status != "completed" || isnothing(stage.output_dir)
            continue
        end

        fig_dir = joinpath(String(stage.output_dir), "figures")
        if !isdir(fig_dir)
            continue
        end

        overlays = filter(p -> endswith(p, "_overlay.csv"), readdir(fig_dir; join=true))
        for overlay_path in sort(overlays)
            push!(all_overlays, (String(stage.name), overlay_path))
        end
    end

    if isempty(all_overlays)
        fig = Dict(
            "data" => Any[],
            "layout" => Dict("title" => "No staged overlay data available for plotting"),
        )
        key_df = DataFrame(series=String[], meaning=String[])
        return fig, key_df
    end

    # Collect traces from all stages
    traces = Any[
        Dict(
            "x" => Float64[],
            "y" => Float64[],
            "mode" => "markers",
            "name" => "Observed (Combined)",
            "marker" => Dict("size" => 6, "color" => "black"),
        ),
    ]

    key_rows = NamedTuple[(series="Observed (Combined)", meaning="Observed data points from all stages")]
    observed_added = false
    pred_models = Set{String}()

    # Process each stage's overlay
    for (stage_name, overlay_path) in all_overlays
        try
            overlay_df = CSV.read(overlay_path, DataFrame)
            if !(:time in Symbol.(names(overlay_df))) || !(:observed in Symbol.(names(overlay_df)))
                continue
            end

            # Append observed data (only add once to marker trace)
            if !observed_added
                traces[1]["x"] = overlay_df.time
                traces[1]["y"] = overlay_df.observed
                observed_added = true
            end

            # Process prediction columns
            pred_cols = [c for c in names(overlay_df) if startswith(String(c), "pred_")]
            for c in pred_cols
                model_name = replace(String(c), "pred_" => "")
                if model_name in pred_models
                    continue  # Already added trace for this model
                end
                push!(pred_models, model_name)

                # Create trace with stage info in name
                push!(traces, Dict(
                    "x" => overlay_df.time,
                    "y" => overlay_df[!, c],
                    "mode" => "lines",
                    "name" => "$(model_name): $(stage_name)",
                    "line" => Dict("width" => 2),
                ))

                # Add to key with LaTeX
                latex_eq = _model_latex(model_name)
                push!(key_rows, (
                    series="$(model_name): $(stage_name)",
                    meaning=latex_eq,
                ))
            end
        catch e
            println("[DEBUG] Error processing overlay $(overlay_path): $(e)")
        end
    end

    # Build figure with all stage data
    fig = Dict(
        "data" => traces,
        "layout" => Dict(
            "title" => "Staged Pipeline Overlay - All Stages",
            "xaxis" => Dict("title" => "Time"),
            "yaxis" => Dict("title" => "Count"),
            "legend" => Dict("orientation" => "v", "x" => 1.0, "y" => 1.0),
            "hovermode" => "closest",
        ),
    )

    key_df = DataFrame(key_rows)
    return fig, key_df
end

function _issue_color(severity::AbstractString)
    if severity == "error"
        return "#9f1239"
    elseif severity == "warning"
        return "#92400e"
    else
        return "#1f2937"
    end
end

function _preflight_panel(path::AbstractString)
    df = _safe_load(path)
    stage_capable = _has_stage_metadata(df)
    report = stage_capable ? preflight_data_quality(df; stages=default_stages()) : preflight_data_quality(df)

    issues = if nrow(report.issues) == 0
        [html_li("No preflight issues detected.")]
    else
        [
            html_li(
                string(r.severity, " | ", r.scope, " | ", r.code, " | ", r.detail, " | Recommendation: ", r.recommendation);
                style=Dict("color" => _issue_color(r.severity), "marginBottom" => "6px"),
            ) for r in eachrow(report.issues)
        ]
    end

    return html_div([
        html_h4("Preflight Summary"),
        !stage_capable ? html_p("Stage coverage checks were skipped because this dataset does not include staged workflow metadata such as culture_type and population_type.") : nothing,
        _as_table(report.summary; limit=20),
        html_h4("Condition Quality"),
        _as_table(report.condition_quality; limit=20),
        html_h4("Stage Coverage"),
        _as_table(report.stage_coverage; limit=20),
        html_h4("Issues"),
        html_ul(issues),
    ])
end

function _conditions_panel(path::AbstractString)
    df = _safe_load(path)
    conditions = build_conditions(df)
    rows = DataFrame(
        condition=[c.name for c in conditions],
        n_points=[length(c.time) for c in conditions],
        t_min=[minimum(c.time) for c in conditions],
        t_max=[maximum(c.time) for c in conditions],
    )
    return html_div([
        html_h4("Built Conditions"),
        _as_table(rows; limit=30),
    ])
end

function _rank_and_plot(path::AbstractString, model_txt::AbstractString, cond_name::AbstractString, n_starts::Int, maxiters::Int)
    df = _safe_load(path)
    conditions = build_conditions(df)
    isempty(conditions) && return html_div([
        html_h4("Model Ranking"),
        html_p("No conditions were built from this dataset. Run Step 1 and review preflight issues."),
    ]), Dict("data" => Any[], "layout" => Dict("title" => "No conditions available for plotting"))

    models = _parse_model_list(model_txt)
    isempty(models) && (models = ["logistic_growth", "gompertz_growth"])

    ranked = rank_models(models, conditions; n_starts=n_starts, maxiters=maxiters, top_k=min(length(models), 5), seed=42)

    successful_models = [
        m for m in ranked.ranking.model
        if haskey(ranked.fits, m) && any(pc -> pc.success && !isempty(pc.observed) && all(isfinite, pc.observed), ranked.fits[m].per_condition)
    ]

    isempty(successful_models) && return html_div([
        html_h4("Model Ranking"),
        html_p("No successful model fits are available for plotting. Check failure log and preflight recommendations."),
        _as_table(ranked.ranking; limit=20),
        html_h4("Failure Log"),
        _as_table(ranked.failures; limit=20),
    ]), Dict("data" => Any[], "layout" => Dict("title" => "No successful predictions available"))

    requested_condition = strip(String(cond_name))
    selected_condition = if isempty(requested_condition)
        conditions[1].name
    else
        idx = _find_condition_index(conditions, requested_condition)
        isnothing(idx) ? conditions[1].name : conditions[idx].name
    end

    selected_model = successful_models[1]
    fit_info = ranked.fits[selected_model]

    cond_idx = _find_condition_index(conditions, selected_condition)
    hit = findfirst(pc -> _condition_match_key(pc.condition) == _condition_match_key(selected_condition) && pc.success, fit_info.per_condition)

    messages = String[]
    if !isempty(requested_condition)
        requested_idx = _find_condition_index(conditions, requested_condition)
        if isnothing(requested_idx)
            push!(messages, "Requested condition was not found. Showing the first available condition instead.")
        end
    end

    if isnothing(cond_idx) || isnothing(hit)
        ranking_panel = html_div([
            html_h4("Model Ranking"),
            html_p("Runtime note: this step scales roughly with models x grouped conditions x optimization starts. Reduce models or starts first if it feels slow."),
            (html_p(m) for m in messages)...,
            html_p("No successful prediction is available for the selected condition/model combination."),
            _as_table(ranked.ranking; limit=20),
            html_h4("Failure Log"),
            _as_table(ranked.failures; limit=20),
        ])
        fig = Dict(
            "data" => Any[],
            "layout" => Dict("title" => "No prediction available for selected condition/model"),
        )
        return ranking_panel, fig
    end

    cond = conditions[cond_idx]
    pred = Float64.(fit_info.per_condition[hit].observed)
    observed = cond.count
    t = cond.time

    vectors_valid = length(pred) == length(observed) && length(observed) == length(t) && !isempty(pred) && all(isfinite, pred) && all(isfinite, observed) && all(isfinite, t)

    if !vectors_valid
        ranking_panel = html_div([
            html_h4("Model Ranking"),
            html_p("Runtime note: this step scales roughly with models x grouped conditions x optimization starts. Reduce models or starts first if it feels slow."),
            (html_p(m) for m in messages)...,
            html_p("Prediction vectors are invalid for plotting (length mismatch or non-finite values)."),
            _as_table(ranked.ranking; limit=20),
            html_h4("Failure Log"),
            _as_table(ranked.failures; limit=20),
            _analysis_panel(ranked, conditions, selected_condition, selected_model),
        ])
        fig = Dict(
            "data" => Any[],
            "layout" => Dict("title" => "Prediction data is invalid for plotting"),
        )
        return ranking_panel, fig
    end
    fig = Dict(
        "data" => Any[
            Dict("x" => t, "y" => observed, "mode" => "markers", "name" => "Observed"),
            Dict("x" => t, "y" => pred, "mode" => "lines", "name" => "Predicted"),
        ],
        "layout" => Dict(
            "title" => "Observed vs Predicted | $(selected_model) | $(selected_condition)",
            "xaxis" => Dict("title" => "Time"),
            "yaxis" => Dict("title" => "Count"),
        ),
    )

    ranking_panel = html_div([
        html_h4("Model Ranking"),
        html_p("Runtime note: this step scales roughly with models x grouped conditions x optimization starts. Reduce models or starts first if it feels slow."),
        (html_p(m) for m in messages)...,
        _as_table(ranked.ranking; limit=20),
        html_h4("Failure Log"),
        _as_table(ranked.failures; limit=20),
        _analysis_panel(ranked, conditions, selected_condition, selected_model),
    ])

    return ranking_panel, fig
end

function _pipeline_panel(path::AbstractString, model_txt::AbstractString)
    df = _safe_load(path)
    cfg = default_config(output_dir="results/gui_pipeline")
    models = _parse_model_list(model_txt)

    run = run_pipeline(
        df;
        config=cfg,
        include_models=models,
        strict_schema=false,
        qc_before_fit=true,
        preflight_before_fit=true,
    )

    ranking_preview = !isnothing(run.exports) && haskey(run.exports, :ranking) ? _read_csv_or_empty(run.exports.ranking) : run.ranking
    summary_preview = !isnothing(run.exports) && haskey(run.exports, :summary) ? _read_csv_or_empty(run.exports.summary) : DataFrame()
    preflight_issues = !isnothing(run.preflight_report) ? run.preflight_report.issues : DataFrame()
    qc_issues = !isnothing(run.qc_report) ? run.qc_report.issues : DataFrame()
    artifact_rows = NamedTuple[]
    if !isnothing(run.exports)
        for (k, v) in pairs(run.exports)
            push!(artifact_rows, (artifact=String(k), path=String(v), exists=ispath(String(v))))
        end
    end
    artifact_df = DataFrame(artifact_rows)
    pipeline_fig, key_df = _pipeline_figure_and_key(run)

    panel = html_div([
        html_h4("Pipeline Result"),
        html_p("Rows in ranking: $(nrow(run.ranking))"),
        html_p("Failures: $(nrow(run.failures))"),
        html_p("Preflight ready: $(run.preflight_report.ready_for_fit)"),
        html_h5("Live Ranking"),
        _as_table(run.ranking; limit=20),
        html_h5("Exported Ranking Preview"),
        _as_table(ranking_preview; limit=20),
        html_h5("Best Model Summary"),
        _as_table(summary_preview; limit=20),
        html_h5("Preflight Issues"),
        _as_table(preflight_issues; limit=20),
        html_h5("QC Issues"),
        _as_table(qc_issues; limit=20),
        html_h5("Pipeline Visualization Key"),
        _as_table(key_df; limit=20),
        html_h5("Generated Artifacts"),
        _as_table(artifact_df; limit=50),
        html_h4("Pipeline Failures"),
        _as_table(run.failures; limit=20),
    ])

    return panel, pipeline_fig
end

function _staged_panel(path::AbstractString)
    println("[DEBUG] _staged_panel: Starting with path=$(path)")
    df = _safe_load(path)
    println("[DEBUG] _staged_panel: Loaded $(nrow(df)) rows from CSV")
    
    if !_has_stage_metadata(df)
        println("[DEBUG] _staged_panel: No stage metadata found")
        panel = html_div([
            html_h4("Staged Pipeline Result"),
            html_p("This dataset is not ready for default staged workflows."),
            html_p("Required staged metadata includes culture_type and population_type."),
            html_p("Use staged_monoculture.csv or coculture_stages.csv, or provide your own staged dataset with those columns."),
        ])
        fig = Dict(
            "data" => Any[],
            "layout" => Dict("title" => "No staged plot available for this dataset"),
        )
        return panel, fig
    end

    println("[DEBUG] _staged_panel: Running staged pipeline...")
    cfg = default_config(output_dir="results/gui_staged")
    run = run_staged_pipeline(
        df;
        stages=default_stages(),
        config=cfg,
        selection_mode=:best_bic,
        strict_schema=false,
        qc_before_fit=true,
        preflight_before_fit=true,
        export_stage_results=true,
    )

    println("[DEBUG] _staged_panel: Staged pipeline completed. Completed=$(run.completed), Halted=$(run.halted_stage)")

    stage_rows = DataFrame(
        stage=[s.name for s in run.stages],
        status=[s.status == "skipped" ? "not_applicable" : s.status for s in run.stages],
        n_conditions=[s.n_conditions for s in run.stages],
        selected_model=[isnothing(s.selected_model) ? "" : s.selected_model for s in run.stages],
    )

    println("[DEBUG] _staged_panel: Building figures and tables...")
    staged_fig, staged_key = _staged_figure_and_key(run)
    println("[DEBUG] _staged_panel: staged_fig type=$(typeof(staged_fig)), has data=$(haskey(staged_fig, "data"))")
    
    staged_rankings = _staged_stage_ranking_table(run)
    best_by_condition = _staged_condition_best_models(run)

    panel = html_div([
        html_h4("Staged Pipeline Result"),
        html_p("Completed: $(run.completed)"),
        html_p("Halted stage: $(isnothing(run.halted_stage) ? "none" : run.halted_stage)"),
        html_p("Manifest: $(isnothing(run.manifest_path) ? "none" : run.manifest_path)"),
        html_h5("Stage Status"),
        _as_table(stage_rows; limit=20),
        html_h5("Best Model By Condition"),
        _as_table(best_by_condition; limit=100),
        html_h5("Model Ranking By Stage"),
        _as_table(staged_rankings; limit=100),
        html_h5("Staged Visualization Key"),
        _as_table(staged_key; limit=20),
        html_h4("Staged Failure Log"),
        _as_table(run.failures; limit=20),
    ])
    println("[DEBUG] _staged_panel: Returning panel and figure")
    return panel, staged_fig
end

app = dash(external_stylesheets=["https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css"])

app.layout = html_main([
    html_h2("GrowthParameterEstimation Pipeline GUI"),
    html_p("Interactive web interface for preflight checks, condition building, ranking, model-vs-data visualization, and full pipeline execution."),
    _timing_guidance(),

    html_div([
        html_label("CSV file path"),
        dcc_input(id="csv-path", type="text", value="", debounce=true, style=Dict("width" => "100%")),
    ]),

    html_div([
        html_label("Model list (comma-separated)"),
        dcc_input(id="models", type="text", value="logistic_growth, gompertz_growth", debounce=true, style=Dict("width" => "100%")),
    ]),

    html_div([
        html_label("Condition name for plotting (optional)"),
        dcc_input(id="condition-name", type="text", value="", debounce=true, style=Dict("width" => "100%")),
    ]),

    html_div([
        html_label("Optimization starts"),
        dcc_input(id="n-starts", type="number", value=8, min=1, max=100),
        html_label("Max iterations"),
        dcc_input(id="maxiters", type="number", value=300, min=20, max=5000),
    ]; style=Dict("display" => "grid", "gridTemplateColumns" => "1fr 1fr 1fr 1fr", "gap" => "12px")),

    html_hr(),

    html_div([
        html_button("Step 1: Run Preflight", id="btn-preflight", n_clicks=0),
        html_button("Step 2: Build Conditions", id="btn-conditions", n_clicks=0),
        html_button("Step 3: Rank Models + Plot", id="btn-rank", n_clicks=0),
        html_button("Step 4: Run Full Pipeline", id="btn-pipeline", n_clicks=0),
        html_button("Step 5: Run Staged Pipeline", id="btn-staged", n_clicks=0),
    ]; style=Dict("display" => "grid", "gridTemplateColumns" => "repeat(5, 1fr)", "gap" => "10px")),

    html_hr(),

    html_section([
        html_h3("Step Output"),
        dcc_loading(
            type="circle",
            color="#0f766e",
            children=html_div(id="step-output", children="Provide a CSV path, then run a step."),
        ),
            "name" => model_name,

    html_section([
        html_h3("Model vs Data Plot"),
            id="fit-plot-loading",
            type="default",
            series=model_name,
            children=dcc_graph(id="fit-plot", figure=Dict("data" => Any[], "layout" => Dict("title" => "Run Step 3 to render model fit"))),
        ),
    ]),
]; style=Dict("maxWidth" => "1300px", "margin" => "0 auto", "padding" => "24px"))

    app,
    Output("step-output", "children"),
    Output("fit-plot", "figure"),
        "title" => "Combined Staged Overlay",
    Input("btn-conditions", "n_clicks"),
    Input("btn-rank", "n_clicks"),
    Input("btn-pipeline", "n_clicks"),
    Input("btn-staged", "n_clicks"),
    State("csv-path", "value"),
    State("models", "value"),
    State("condition-name", "value"),
    State("n-starts", "value"),
    State("maxiters", "value"),
) do n_pre, n_cond, n_rank, n_pipe, n_staged, csv_path, models, cond_name, n_starts, maxiters
    base_fig = Dict("data" => Any[], "layout" => Dict("title" => "Run Step 3 to render model fit"))
function _staged_figure_for_stage(run, stage_name::AbstractString)
    overlay_path = nothing
    
    for stage in run.stages
        if String(stage.name) != stage_name
            continue
        end
        if stage.status != "completed" || isnothing(stage.output_dir)
            break
        end

        fig_dir = joinpath(String(stage.output_dir), "figures")
        if !isdir(fig_dir)
            break
        end

        overlays = filter(p -> endswith(p, "_overlay.csv"), readdir(fig_dir; join=true))
        if !isempty(overlays)
            overlay_path = first(sort(overlays))
        end
        break
    end

    if isnothing(overlay_path)
        return Dict(
            "data" => Any[],
            "layout" => Dict("title" => "No overlay data available for stage: $(stage_name)"),
        ), DataFrame(series=String[], meaning=String[])
    end

    overlay_df = CSV.read(overlay_path, DataFrame)
    if !(:time in Symbol.(names(overlay_df))) || !(:observed in Symbol.(names(overlay_df)))
        return Dict(
            "data" => Any[],
            "layout" => Dict("title" => "Stage overlay CSV missing required columns"),
        ), DataFrame(series=["observed"], meaning=["Observed data points"])
    end

    traces = Any[
        Dict(
            "x" => overlay_df.time,
            "y" => overlay_df.observed,
            "mode" => "markers",
            "name" => "Observed",
            "marker" => Dict("size" => 8, "color" => "black"),
        ),
    ]

    key_rows = NamedTuple[(series="Observed", meaning="Observed data points from input CSV")]
    pred_cols = [c for c in names(overlay_df) if startswith(String(c), "pred_")]
    for c in sort(pred_cols)
        model_name = replace(String(c), "pred_" => "")
        push!(traces, Dict(
            "x" => overlay_df.time,
            "y" => overlay_df[!, c],
            "mode" => "lines",
            "name" => model_name,
            "line" => Dict("width" => 2),
        ))

        latex_eq = _model_latex(model_name)
        push!(key_rows, (series=model_name, meaning=latex_eq))
    end

    fig = Dict(
        "data" => traces,
        "layout" => Dict(
            "title" => "Stage: $(stage_name)",
            "xaxis" => Dict("title" => "Time"),
            "yaxis" => Dict("title" => "Count"),
            "legend" => Dict("orientation" => "v", "x" => 1.0, "y" => 1.0),
            "hovermode" => "closest",
        ),
    )

    return fig, DataFrame(key_rows)
end

    if isnothing(csv_path) || isempty(strip(String(csv_path)))
        return html_p("Please provide a CSV path first."), base_fig
    end

    path = String(csv_path)
    triggered_id = _triggered_step_id()
    println("[DEBUG] Callback triggered: triggered_id=$(triggered_id)")

    try
        if triggered_id == "btn-preflight"
            println("[DEBUG] Running preflight...")
            panel = _preflight_panel(path)
            return panel, base_fig
        elseif triggered_id == "btn-conditions"
            println("[DEBUG] Running conditions...")
            panel = _conditions_panel(path)
            return panel, base_fig
        elseif triggered_id == "btn-rank"
            println("[DEBUG] Running rank...")
            panel, fig = _rank_and_plot(path, String(models), String(cond_name), Int(n_starts), Int(maxiters))
            return panel, fig
        elseif triggered_id == "btn-pipeline"
            println("[DEBUG] Running pipeline...")
            panel, fig = _pipeline_panel(path, String(models))
            return panel, fig
        elseif triggered_id == "btn-staged"
            println("[DEBUG] Running staged...")
            panel, fig = _staged_panel(path)
            println("[DEBUG] Callback: Got panel type=$(typeof(panel)), fig type=$(typeof(fig))")
            return panel, fig
        else
            println("[DEBUG] No matching step ID")
            return html_p("Choose a step to run."), base_fig
        end
    catch err
        println("[DEBUG] ERROR in callback: $(err)")
        msg = sprint(showerror, err)
        return html_div([
            html_h4("Step failed"),
            html_p(msg),
            html_p("Tip: run Step 1 first and check preflight issues/recommendations."),
        ]), base_fig
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Pipeline GUI running at http://$(HOST):$(PORT)")
    try
        run_server(app, HOST, PORT; debug=false)
    catch err
        msg = sprint(showerror, err)
        if occursin("EADDRINUSE", msg)
            println("Port $(PORT) is already in use. Set GPE_GUI_PORT to another port, for example:")
            println("GPE_GUI_PORT=8051 julia --project=examples/gui examples/gui/pipeline_gui_app.jl")
        end
        rethrow(err)
    end
end
