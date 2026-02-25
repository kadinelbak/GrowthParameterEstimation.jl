module DataLayer

using CSV
using DataFrames

export REQUIRED_COLUMNS, load_timeseries, normalize_schema, validate_timeseries

const REQUIRED_COLUMNS = [:time, :count, :error, :dose, :cell_line, :density, :replicate, :unit_time, :unit_count]

function load_timeseries(path::AbstractString; kwargs...)
    df = CSV.read(path, DataFrame; kwargs...)
    return normalize_schema(df)
end

function normalize_schema(
    df::DataFrame;
    column_map::Dict{Symbol,Symbol} = Dict{Symbol,Symbol}(),
    defaults::Dict{Symbol,Any} = Dict(
        :error => missing,
        :dose => 0.0,
        :cell_line => "unknown",
        :density => missing,
        :replicate => 1,
        :unit_time => "h",
        :unit_count => "count",
    ),
)
    work = copy(df)
    current_names = Set(Symbol.(names(work)))

    for (from, to) in column_map
        if from in current_names
            rename!(work, from => to)
            delete!(current_names, from)
            push!(current_names, to)
        end
    end

    for col in REQUIRED_COLUMNS
        if !(col in current_names)
            default_value = get(defaults, col, missing)
            work[!, col] = fill(default_value, nrow(work))
            push!(current_names, col)
        end
    end

    work = work[:, REQUIRED_COLUMNS]

    work.time = Float64.(work.time)
    work.count = Float64.(work.count)

    if all(ismissing, work.error)
        work.error = fill(1.0, nrow(work))
    else
        work.error = [ismissing(v) ? 1.0 : Float64(v) for v in work.error]
    end

    work.dose = [ismissing(v) ? 0.0 : Float64(v) for v in work.dose]
    work.replicate = [ismissing(v) ? 1 : Int(v) for v in work.replicate]

    return work
end

function validate_timeseries(df::DataFrame)
    missing_cols = setdiff(REQUIRED_COLUMNS, Symbol.(names(df)))
    if !isempty(missing_cols)
        error("Missing required columns: $(join(string.(missing_cols), ", "))")
    end

    if any(!isfinite(v) for v in df.time) || any(!isfinite(v) for v in df.count)
        error("Columns :time and :count must be finite numeric values")
    end

    if any(v < 0 for v in df.count)
        error("Column :count must be nonnegative")
    end

    grouped = groupby(df, [:dose, :cell_line, :density, :replicate])
    for g in grouped
        t = collect(g.time)
        if any(diff(t) .< 0)
            error("Time must be monotone within each condition")
        end
    end

    if any(v <= 0 for v in df.error)
        error("Column :error must be positive")
    end

    return true
end

end # module DataLayer
