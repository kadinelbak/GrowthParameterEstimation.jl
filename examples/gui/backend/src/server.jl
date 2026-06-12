module GUIBackend

using CSV
using DataFrames
using Dates
using HTTP
using JSON3
using Tables

const DEFAULT_PORT = 8050
const MAX_PREVIEW_ROWS = 25
const MAX_PLOT_POINTS_PER_FILE = 2000
const BACKEND_DATA_DIR = normpath(joinpath(@__DIR__, "..", "data"))
const CUSTOM_MODELS_STORE = joinpath(BACKEND_DATA_DIR, "custom_models.json")
const CUSTOM_MODELS_REGISTRY_FILE = joinpath(BACKEND_DATA_DIR, "custom_models_registry.jl")

const CUSTOM_MODEL_TEMPLATES = Dict{String,Any}(
    "logistic_basic" => Dict(
        "label" => "Logistic Growth",
        "equation" => "dN/dt = r*N*(1 - N/K)",
        "defaultPlainMath" => "d(N,t) = r*N*(1 - frac(N,K))",
        "defaultRhs" => "r*N*(1 - N/max(K, 1e-8))",
        "defaultMathTex" => "\\frac{dN}{dt} = rN\\left(1 - \\frac{N}{K}\\right)",
        "paramNames" => ["r", "K"],
        "defaultBounds" => [[1e-6, 5.0], [1e-3, 1e7]],
    ),
    "logistic_linear_kill" => Dict(
        "label" => "Logistic + Linear Kill",
        "equation" => "dN/dt = r*N*(1 - N/K) - kill_coeff*dose*N",
        "defaultPlainMath" => "d(N,t) = r*N*(1 - frac(N,K)) - kill_coeff*dose*N",
        "defaultRhs" => "r*N*(1 - N/max(K, 1e-8)) - kill_coeff*dose*N",
        "defaultMathTex" => "\\frac{dN}{dt} = rN\\left(1 - \\frac{N}{K}\\right) - k_{kill}DN",
        "paramNames" => ["r", "K", "kill_coeff"],
        "defaultBounds" => [[1e-6, 5.0], [1e-3, 1e7], [0.0, 5.0]],
    ),
    "theta_logistic_hill_inhibition" => Dict(
        "label" => "Theta Logistic + Hill Inhibition",
        "equation" => "dN/dt = (1 - I(dose))*r*N*(1 - (N/K)^theta)",
        "defaultPlainMath" => "d(N,t) = (1 - frac(pow(dose,hill), pow(ic50,hill) + pow(dose,hill)))*r*N*(1 - pow(frac(N,K),theta))",
        "defaultRhs" => "max(1e-12, 1 - (dose^hill/(ic50^hill + dose^hill + 1e-12))) * r*N*(1 - (N/max(K, 1e-8))^theta)",
        "defaultMathTex" => "\\frac{dN}{dt} = (1-I(D))rN\\left(1-\\left(\\frac{N}{K}\\right)^\\theta\\right)",
        "paramNames" => ["r", "K", "theta", "ic50", "hill"],
        "defaultBounds" => [[1e-6, 5.0], [1e-3, 1e7], [0.1, 5.0], [1e-6, 1e3], [0.1, 5.0]],
    ),
    "theta_logistic_hill_kill" => Dict(
        "label" => "Theta Logistic + Hill Kill",
        "equation" => "dN/dt = r*N*(1 - (N/K)^theta) - K(dose)*N",
        "defaultPlainMath" => "d(N,t) = r*N*(1 - pow(frac(N,K),theta)) - emax_kill*frac(pow(dose,hill), pow(ic50,hill) + pow(dose,hill))*N",
        "defaultRhs" => "r*N*(1 - (N/max(K, 1e-8))^theta) - (emax_kill*(dose^hill/(ic50^hill + dose^hill + 1e-12)))*N",
        "defaultMathTex" => "\\frac{dN}{dt} = rN\\left(1-\\left(\\frac{N}{K}\\right)^\\theta\\right)-K(D)N",
        "paramNames" => ["r", "K", "theta", "emax_kill", "ic50", "hill"],
        "defaultBounds" => [[1e-6, 5.0], [1e-3, 1e7], [0.1, 5.0], [0.0, 5.0], [1e-6, 1e3], [0.1, 5.0]],
    ),
)

const RESERVED_RHS_IDENTIFIERS = Set([
    "du", "u", "p", "t", "exposure", "function", "begin", "end", "for", "while", "if", "else", "elseif", "let", "quote"
])

const RHS_ALLOWED_SYMBOLS = Set([
    "N", "dose", "max", "min", "abs", "exp", "log", "sqrt", "clamp", "e", "pi"
])

function _configured_port()
    raw = get(ENV, "GPE_GUI_BACKEND_PORT", string(DEFAULT_PORT))
    try
        return parse(Int, raw)
    catch
        return DEFAULT_PORT
    end
end

function _ensure_data_dir!()
    isdir(BACKEND_DATA_DIR) || mkpath(BACKEND_DATA_DIR)
    return nothing
end

function _safe_model_symbol(name::AbstractString)
    base = _normalize_name(name)
    if isempty(base)
        return "custom_model"
    end
    if !isletter(base[1])
        base = "m_" * base
    end
    return base
end

function _default_rhs_for_variant(variant::AbstractString)
    if haskey(CUSTOM_MODEL_TEMPLATES, variant)
        return String(CUSTOM_MODEL_TEMPLATES[variant]["defaultRhs"])
    end
    return "r*N*(1 - N/max(K, 1e-8))"
end

function _normalize_loaded_custom_model(item)
    d = Dict{String,Any}(String(k) => v for (k, v) in pairs(item))
    variant = haskey(d, "variant") ? String(d["variant"]) : "logistic_basic"
    if !haskey(d, "rhsEquation")
        d["rhsEquation"] = _default_rhs_for_variant(variant)
    end
    if !haskey(d, "mathText")
        if haskey(CUSTOM_MODEL_TEMPLATES, variant)
            d["mathText"] = String(CUSTOM_MODEL_TEMPLATES[variant]["defaultMathTex"])
        else
            d["mathText"] = ""
        end
    end
    return d
end

function _read_custom_models()
    _ensure_data_dir!()
    if !isfile(CUSTOM_MODELS_STORE)
        return Any[]
    end
    raw = read(CUSTOM_MODELS_STORE, String)
    cleaned = replace(raw, "\ufeff" => "")
    parsed = try
        JSON3.read(cleaned)
    catch
        return Any[]
    end
    if !haskey(parsed, :models)
        return Any[]
    end
    return [_normalize_loaded_custom_model(item) for item in parsed[:models]]
end

function _write_custom_models!(models)
    _ensure_data_dir!()
    write(CUSTOM_MODELS_STORE, JSON3.write(Dict("models" => models), allow_inf = false))
    return nothing
end

function _validate_model_payload(payload)
    haskey(payload, :name) || throw(ArgumentError("Missing model name"))
    haskey(payload, :variant) || throw(ArgumentError("Missing model variant"))
    haskey(payload, :rhsEquation) || throw(ArgumentError("Missing rhsEquation"))
    haskey(payload, :paramNames) || throw(ArgumentError("Missing paramNames"))
    haskey(payload, :bounds) || throw(ArgumentError("Missing parameter bounds"))

    name = strip(String(payload[:name]))
    if isempty(name)
        throw(ArgumentError("Model name cannot be empty"))
    end
    if !occursin(r"^[A-Za-z][A-Za-z0-9_\- ]*$", name)
        throw(ArgumentError("Model name must start with a letter and use only letters, numbers, spaces, underscore, or hyphen"))
    end

    variant = String(payload[:variant])
    haskey(CUSTOM_MODEL_TEMPLATES, variant) || throw(ArgumentError("Unknown model variant: $(variant)"))

    rhs_equation = String(payload[:rhsEquation])
    rhs_clean = strip(rhs_equation)
    isempty(rhs_clean) && throw(ArgumentError("Equation RHS cannot be empty"))
    length(rhs_clean) <= 500 || throw(ArgumentError("Equation RHS is too long"))
    occursin(r"^[A-Za-z0-9_+\-*/^().,\s]+$", rhs_clean) || throw(ArgumentError("Equation RHS contains unsupported characters"))

    param_names = [String(x) for x in collect(payload[:paramNames])]
    length(param_names) >= 1 || throw(ArgumentError("At least one parameter is required"))
    length(Set(param_names)) == length(param_names) || throw(ArgumentError("Parameter names must be unique"))

    param_set = Set(param_names)
    for pname in param_names
        occursin(r"^[A-Za-z][A-Za-z0-9_]*$", pname) || throw(ArgumentError("Invalid parameter name: $(pname)"))
        !(pname in RESERVED_RHS_IDENTIFIERS) || throw(ArgumentError("Reserved parameter name is not allowed: $(pname)"))
    end

    token_matches = collect(eachmatch(r"[A-Za-z_][A-Za-z0-9_]*", rhs_clean))
    for m in token_matches
        token = String(m.match)
        if token in RESERVED_RHS_IDENTIFIERS
            throw(ArgumentError("Equation contains reserved token: $(token)"))
        end
        if !(token in RHS_ALLOWED_SYMBOLS) && !(token in param_set)
            throw(ArgumentError("Equation token $(token) is not declared. Use parameter names or supported symbols (N, dose, max, min, abs, exp, log, sqrt, clamp)."))
        end
    end

    bounds_input = collect(payload[:bounds])
    length(bounds_input) == length(param_names) || throw(ArgumentError("Expected $(length(param_names)) bounds entries"))

    parsed_bounds = Vector{Tuple{Float64,Float64}}()
    for b in bounds_input
        lo = Float64(b[1])
        hi = Float64(b[2])
        lo < hi || throw(ArgumentError("Each bound must satisfy lower < upper"))
        push!(parsed_bounds, (lo, hi))
    end

    description = haskey(payload, :description) ? String(payload[:description]) : ""
    math_text = haskey(payload, :mathText) ? String(payload[:mathText]) : ""
    length(math_text) <= 1000 || throw(ArgumentError("Math text is too long"))

    return Dict(
        "name" => name,
        "variant" => variant,
        "description" => description,
        "mathText" => math_text,
        "rhsEquation" => rhs_clean,
        "paramNames" => param_names,
        "bounds" => [[b[1], b[2]] for b in parsed_bounds],
        "createdAt" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
    )
end

function _model_function_lines(model::Dict{String,Any})
    fn_name = _safe_model_symbol(model["name"]) * "_ode!"
    lines = String[]

    push!(lines, "function $(fn_name)(du, u, p, t, exposure)")
    push!(lines, "    N = max(u[1], 0.0)")

    param_names = [String(x) for x in model["paramNames"]]
    push!(lines, "    $(join(param_names, ", ")) = p")
    push!(lines, "    dose = max(exposure(t), 0.0)")
    rhs = haskey(model, "rhsEquation") ? String(model["rhsEquation"]) : _default_rhs_for_variant(String(model["variant"]))
    push!(lines, "    du[1] = $(rhs)")

    push!(lines, "    return nothing")
    push!(lines, "end")
    return lines, fn_name
end

function _generate_custom_model_registry!(models)
    _ensure_data_dir!()
    lines = String[]

    push!(lines, "module GUICustomModels")
    push!(lines, "")
    push!(lines, "using GrowthParameterEstimation")
    push!(lines, "")

    fn_names = String[]
    for model in models
        fn_lines, fn_name = _model_function_lines(model)
        append!(lines, fn_lines)
        push!(lines, "")
        push!(fn_names, fn_name)
    end

    push!(lines, "function register_custom_models!(; overwrite::Bool = true)")
    for (model, fn_name) in zip(models, fn_names)
        param_symbols = [":" * name for name in model["paramNames"]]
        bounds_str = ["($(b[1]), $(b[2]))" for b in model["bounds"]]

        push!(lines, "    register_model!(ModelSpec(")
        push!(lines, "        name = \"$(model["name"])\",")
        push!(lines, "        ode! = $(fn_name),")
        push!(lines, "        param_names = [$(join(param_symbols, ", "))],")
        push!(lines, "        bounds = [$(join(bounds_str, ", "))],")
        push!(lines, "        n_states = 1,")
        push!(lines, "        observable = u -> u[1],")
        push!(lines, "        base_growth_family = \"custom_logistic\"," )
        push!(lines, "    ); overwrite = overwrite)")
    end
    push!(lines, "    return nothing")
    push!(lines, "end")
    push!(lines, "")
    push!(lines, "end # module")

    write(CUSTOM_MODELS_REGISTRY_FILE, join(lines, "\n") * "\n")
    return CUSTOM_MODELS_REGISTRY_FILE
end

is_numeric_column(col::AbstractVector) = eltype(skipmissing(col)) <: Number

function _normalize_name(name::AbstractString)
    lowered = lowercase(strip(name))
    return replace(lowered, r"[^a-z0-9]+" => "_")
end

function _infer_time_column(df::DataFrame)
    names_raw = String.(names(df))
    names_norm = _normalize_name.(names_raw)

    priority = Set([
        "time", "time_h", "time_hr", "time_hour", "hours", "hour", "day", "days", "t"
    ])

    for (raw, norm) in zip(names_raw, names_norm)
        if norm in priority && is_numeric_column(df[!, raw])
            return raw
        end
    end

    for (raw, norm) in zip(names_raw, names_norm)
        if occursin("time", norm) && is_numeric_column(df[!, raw])
            return raw
        end
    end

    for raw in names_raw
        if is_numeric_column(df[!, raw])
            return raw
        end
    end

    return nothing
end

function _infer_count_column(df::DataFrame, time_col)
    names_raw = String.(names(df))
    names_norm = _normalize_name.(names_raw)

    priority = Set([
        "count", "counts", "cell_count", "cells", "measurement", "value", "y", "response", "signal"
    ])

    for (raw, norm) in zip(names_raw, names_norm)
        if raw != time_col && norm in priority && is_numeric_column(df[!, raw])
            return raw
        end
    end

    for (raw, norm) in zip(names_raw, names_norm)
        if raw != time_col && (occursin("count", norm) || occursin("cell", norm) || occursin("measure", norm)) && is_numeric_column(df[!, raw])
            return raw
        end
    end

    for raw in names_raw
        if raw != time_col && is_numeric_column(df[!, raw])
            return raw
        end
    end

    return nothing
end

function _infer_id_column(df::DataFrame)
    names_raw = String.(names(df))
    names_norm = _normalize_name.(names_raw)
    priority = Set(["filepath", "file_path", "path", "id", "sample_id", "replicate", "condition"])

    for (raw, norm) in zip(names_raw, names_norm)
        if norm in priority
            return raw
        end
    end

    for (raw, norm) in zip(names_raw, names_norm)
        if occursin("id", norm) || occursin("path", norm) || occursin("file", norm)
            return raw
        end
    end

    return nothing
end

function _is_identifier_like(name::AbstractString)
    norm = _normalize_name(name)
    if norm == "id" || endswith(norm, "_id")
        return true
    end
    return occursin("sample", norm) || occursin("file", norm) || occursin("path", norm) || occursin("replicate", norm) || norm == "rep"
end

function _fit_candidate_columns(df::DataFrame, time_col, count_col, id_col)
    blocked = Set(filter(!isnothing, [time_col, count_col, id_col]))
    out = String[]
    for name in String.(names(df))
        if !(name in blocked) && is_numeric_column(df[!, name]) && !_is_identifier_like(name)
            push!(out, name)
        end
    end
    return out
end

function _column_type_name(col::AbstractVector)
    if isempty(col)
        return "Unknown"
    end
    return string(eltype(col))
end

function _to_plot_points(df::DataFrame, time_col, count_col, id_col)
    if isnothing(time_col) || isnothing(count_col)
        return Any[]
    end

    n = nrow(df)
    if n == 0
        return Any[]
    end

    step = max(1, ceil(Int, n / MAX_PLOT_POINTS_PER_FILE))
    points = Any[]

    for i in 1:step:n
        row = df[i, :]
        x = row[time_col]
        y = row[count_col]
        if ismissing(x) || ismissing(y)
            continue
        end
        point = Dict{String, Any}(
            "rowIndex" => i,
            "x" => x,
            "y" => y,
        )
        if !isnothing(id_col)
            point["id"] = row[id_col]
        end
        push!(points, point)
    end

    return points
end

function _preview(df::DataFrame)
    n = min(MAX_PREVIEW_ROWS, nrow(df))
    if n == 0
        return Any[]
    end
    sub = df[1:n, :]
    return [Dict(string(k) => v for (k, v) in pairs(Tables.rowtable(sub)[i])) for i in 1:n]
end

function _inspect_one_file(payload)
    id = haskey(payload, :id) ? String(payload[:id]) : string(rand(UInt))
    name = haskey(payload, :name) ? String(payload[:name]) : "unnamed.csv"

    if !haskey(payload, :content)
        return Dict(
            "id" => id,
            "name" => name,
            "ok" => false,
            "error" => "Missing content field in file payload",
        )
    end

    content = String(payload[:content])
    df = CSV.read(IOBuffer(content), DataFrame)

    names_raw = String.(names(df))
    time_col = _infer_time_column(df)
    count_col = _infer_count_column(df, time_col)
    id_col = _infer_id_column(df)
    fit_candidates = _fit_candidate_columns(df, time_col, count_col, id_col)

    col_stats = Any[]
    for col_name in names_raw
        col = df[!, col_name]
        push!(col_stats, Dict(
            "name" => col_name,
            "type" => _column_type_name(col),
            "missingCount" => count(ismissing, col),
            "nonMissingCount" => count(x -> !ismissing(x), col),
        ))
    end

    warnings = String[]
    if isnothing(time_col)
        push!(warnings, "Could not confidently infer a time column")
    end
    if isnothing(count_col)
        push!(warnings, "Could not confidently infer a count column")
    end

    return Dict(
        "id" => id,
        "name" => name,
        "ok" => true,
        "nRows" => nrow(df),
        "nCols" => ncol(df),
        "columns" => col_stats,
        "columnSuggestions" => Dict(
            "time" => time_col,
            "count" => count_col,
            "id" => id_col,
            "fitCandidates" => fit_candidates,
        ),
        "preview" => _preview(df),
        "plotSeries" => _to_plot_points(df, time_col, count_col, id_col),
        "warnings" => warnings,
    )
end

function _cors_headers()
    return [
        "Access-Control-Allow-Origin" => "*",
        "Access-Control-Allow-Methods" => "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers" => "Content-Type",
    ]
end

function _json_response(status::Integer, body)
    payload = JSON3.write(body)
    headers = vcat(["Content-Type" => "application/json"], _cors_headers())
    return HTTP.Response(status, headers, payload)
end

function _route(req::HTTP.Request)
    if req.method == "OPTIONS"
        return HTTP.Response(204, _cors_headers())
    end

    if req.method == "GET" && req.target == "/health"
        return _json_response(200, Dict("ok" => true, "service" => "gpe-gui-backend"))
    end

    if req.method == "GET" && req.target == "/api/models/templates"
        return _json_response(200, Dict("ok" => true, "templates" => CUSTOM_MODEL_TEMPLATES))
    end

    if req.method == "GET" && req.target == "/api/models/custom"
        models = _read_custom_models()
        return _json_response(200, Dict(
            "ok" => true,
            "models" => models,
            "registryFile" => CUSTOM_MODELS_REGISTRY_FILE,
            "loadHint" => "using GrowthParameterEstimation; register_models_from_file!(\"$(CUSTOM_MODELS_REGISTRY_FILE)\")",
        ))
    end

    if req.method == "POST" && req.target == "/api/models/custom"
        body = try
            JSON3.read(String(req.body))
        catch err
            return _json_response(400, Dict("ok" => false, "error" => "Invalid JSON body", "details" => sprint(showerror, err)))
        end

        parsed = try
            _validate_model_payload(body)
        catch err
            return _json_response(400, Dict("ok" => false, "error" => sprint(showerror, err)))
        end

        models = _read_custom_models()
        model_name = lowercase(parsed["name"])
        if any(lowercase(String(item["name"])) == model_name for item in models)
            return _json_response(409, Dict("ok" => false, "error" => "Model name already exists"))
        end

        push!(models, parsed)
        _write_custom_models!(models)
        registry_file = _generate_custom_model_registry!(models)

        return _json_response(200, Dict(
            "ok" => true,
            "model" => parsed,
            "models" => models,
            "registryFile" => registry_file,
            "message" => "Custom model saved. Use register_models_from_file! to load into the fitter.",
        ))
    end

    if req.method == "POST" && req.target == "/api/csv/inspect"
        body = try
            JSON3.read(String(req.body))
        catch err
            return _json_response(400, Dict("ok" => false, "error" => "Invalid JSON body", "details" => sprint(showerror, err)))
        end

        if !haskey(body, :files)
            return _json_response(400, Dict("ok" => false, "error" => "Expected body.files"))
        end

        inspected = Any[]
        for file_payload in body[:files]
            try
                push!(inspected, _inspect_one_file(file_payload))
            catch err
                push!(inspected, Dict(
                    "id" => haskey(file_payload, :id) ? String(file_payload[:id]) : string(rand(UInt)),
                    "name" => haskey(file_payload, :name) ? String(file_payload[:name]) : "unnamed.csv",
                    "ok" => false,
                    "error" => sprint(showerror, err),
                ))
            end
        end

        total_rows = sum(get(item, "nRows", 0) for item in inspected if get(item, "ok", false))
        return _json_response(200, Dict(
            "ok" => true,
            "files" => inspected,
            "summary" => Dict(
                "fileCount" => length(inspected),
                "successfulFiles" => count(get(item, "ok", false) for item in inspected),
                "totalRows" => total_rows,
            ),
        ))
    end

    return _json_response(404, Dict("ok" => false, "error" => "Route not found"))
end

function run_server(; host::AbstractString = "127.0.0.1", port::Integer = DEFAULT_PORT)
    _generate_custom_model_registry!(_read_custom_models())
    println("Starting GPE GUI backend on http://$(host):$(port)")
    HTTP.serve(_route, host, port)
end

end # module

using .GUIBackend
GUIBackend.run_server(port = GUIBackend._configured_port())
