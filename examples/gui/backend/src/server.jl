module GUIBackend

using CSV
using DataFrames
using HTTP
using JSON3
using Tables

const DEFAULT_PORT = 8050
const MAX_PREVIEW_ROWS = 25
const MAX_PLOT_POINTS_PER_FILE = 2000

function _configured_port()
    raw = get(ENV, "GPE_GUI_BACKEND_PORT", string(DEFAULT_PORT))
    try
        return parse(Int, raw)
    catch
        return DEFAULT_PORT
    end
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

function _fit_candidate_columns(df::DataFrame, time_col, count_col, id_col)
    blocked = Set(filter(!isnothing, [time_col, count_col, id_col]))
    out = String[]
    for name in String.(names(df))
        if !(name in blocked) && is_numeric_column(df[!, name])
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
    println("Starting GPE GUI backend on http://$(host):$(port)")
    HTTP.serve(_route, host, port)
end

end # module

using .GUIBackend
GUIBackend.run_server(port = GUIBackend._configured_port())
