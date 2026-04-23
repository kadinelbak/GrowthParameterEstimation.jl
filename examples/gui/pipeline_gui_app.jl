using Dash
using CSV
using DataFrames
using GrowthParameterEstimation

const HOST = "127.0.0.1"
const PORT = 8050

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
    models = _parse_model_list(model_txt)
    isempty(models) && (models = ["logistic_growth", "gompertz_growth"])

    ranked = rank_models(models, conditions; n_starts=n_starts, maxiters=maxiters, top_k=min(length(models), 5), seed=42)

    selected_condition = isempty(cond_name) ? conditions[1].name : cond_name
    selected_model = haskey(ranked.fits, ranked.ranking.model[1]) ? ranked.ranking.model[1] : first(keys(ranked.fits))

    pred = nothing
    observed = nothing
    t = nothing

    if haskey(ranked.fits, selected_model)
        fit_info = ranked.fits[selected_model]
        hit = findfirst(pc -> pc.condition == selected_condition, fit_info.per_condition)
        if !isnothing(hit)
            observed = conditions[findfirst(c -> c.name == selected_condition, conditions)].count
            t = conditions[findfirst(c -> c.name == selected_condition, conditions)].time
            pred = fit_info.per_condition[hit].observed
        end
    end

    fig = if isnothing(pred)
        Dict(
            "data" => Any[],
            "layout" => Dict("title" => "No prediction available for selected condition/model"),
        )
    else
        Dict(
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
    end

    ranking_panel = html_div([
        html_h4("Model Ranking"),
        html_p("Runtime note: this step scales roughly with models x grouped conditions x optimization starts. Reduce models or starts first if it feels slow."),
        _as_table(ranked.ranking; limit=20),
        html_h4("Failure Log"),
        _as_table(ranked.failures; limit=20),
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

    return html_div([
        html_h4("Pipeline Result"),
        html_p("Rows in ranking: $(nrow(run.ranking))"),
        html_p("Failures: $(nrow(run.failures))"),
        html_p("Preflight ready: $(run.preflight_report.ready_for_fit)"),
        _as_table(run.ranking; limit=20),
        html_h4("Pipeline Failures"),
        _as_table(run.failures; limit=20),
    ])
end

function _staged_panel(path::AbstractString)
    df = _safe_load(path)
    if !_has_stage_metadata(df)
        return html_div([
            html_h4("Staged Pipeline Result"),
            html_p("This dataset is not ready for default staged workflows."),
            html_p("Required staged metadata includes culture_type and population_type."),
            html_p("Use staged_monoculture.csv or coculture_stages.csv, or provide your own staged dataset with those columns."),
        ])
    end

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

    stage_rows = DataFrame(
        stage=[s.name for s in run.stages],
        status=[s.status for s in run.stages],
        n_conditions=[s.n_conditions for s in run.stages],
        selected_model=[isnothing(s.selected_model) ? "" : s.selected_model for s in run.stages],
    )

    return html_div([
        html_h4("Staged Pipeline Result"),
        html_p("Completed: $(run.completed)"),
        html_p("Halted stage: $(isnothing(run.halted_stage) ? "none" : run.halted_stage)"),
        html_p("Manifest: $(isnothing(run.manifest_path) ? "none" : run.manifest_path)"),
        _as_table(stage_rows; limit=20),
        html_h4("Staged Failure Log"),
        _as_table(run.failures; limit=20),
    ])
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
            id="step-output-loading",
            type="circle",
            color="#0f766e",
            children=html_div(id="step-output", children="Provide a CSV path, then run a step."),
        ),
    ]),

    html_section([
        html_h3("Model vs Data Plot"),
        dcc_loading(
            id="fit-plot-loading",
            type="default",
            color="#0f766e",
            children=dcc_graph(id="fit-plot", figure=Dict("data" => Any[], "layout" => Dict("title" => "Run Step 3 to render model fit"))),
        ),
    ]),
]; style=Dict("maxWidth" => "1300px", "margin" => "0 auto", "padding" => "24px"))

callback!(
    app,
    Output("step-output", "children"),
    Output("fit-plot", "figure"),
    Input("btn-preflight", "n_clicks"),
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

    if isnothing(csv_path) || isempty(strip(String(csv_path)))
        return html_p("Please provide a CSV path first."), base_fig
    end

    path = String(csv_path)
    steps = [Int(n_pre), Int(n_cond), Int(n_rank), Int(n_pipe), Int(n_staged)]
    step_idx = argmax(steps)

    try
        if step_idx == 1
            panel = _preflight_panel(path)
            return panel, base_fig
        elseif step_idx == 2
            panel = _conditions_panel(path)
            return panel, base_fig
        elseif step_idx == 3
            panel, fig = _rank_and_plot(path, String(models), String(cond_name), Int(n_starts), Int(maxiters))
            return panel, fig
        elseif step_idx == 4
            panel = _pipeline_panel(path, String(models))
            return panel, base_fig
        elseif step_idx == 5
            panel = _staged_panel(path)
            return panel, base_fig
        else
            return html_p("Choose a step to run."), base_fig
        end
    catch err
        msg = sprint(showerror, err)
        return html_div([
            html_h4("Step failed"),
            html_p(msg),
            html_p("Tip: run Step 1 first and check preflight issues/recommendations."),
        ]), base_fig
    end
end

println("Pipeline GUI running at http://$(HOST):$(PORT)")
run_server(app, HOST, PORT; debug=false)
