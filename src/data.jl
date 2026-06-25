module DataLayer

using CSV
using DataFrames

export REQUIRED_COLUMNS, STRICT_REQUIRED_METADATA, load_timeseries, normalize_schema,
       validate_timeseries, validate_required_metadata

"""
    REQUIRED_COLUMNS

Vector of column names that are required in every timeseries DataFrame.
These columns must be present for the data to be considered valid.
"""
const REQUIRED_COLUMNS = [:time, :count, :error, :dose, :treatment_amount, :cell_line, :density, :replicate, :unit_time, :unit_count]

"""
    STRICT_REQUIRED_METADATA

Vector of column names that are required for strict schema validation.
In strict mode, these metadata columns must be present and contain valid values.
"""
const STRICT_REQUIRED_METADATA = [:culture_type, :population_type, :cell_line, :treatment_amount, :density]

function _as_float_or_missing(v)
    if ismissing(v)
        return missing
    elseif v isa AbstractString
        s = strip(String(v))
        isempty(s) && return missing
        parsed = tryparse(Float64, s)
        return isnothing(parsed) ? missing : parsed
    end

    try
        return Float64(v)
    catch
        return missing
    end
end

function _as_int_or_default(v, default::Int = 1)
    if ismissing(v)
        return default
    elseif v isa Integer
        return Int(v)
    elseif v isa AbstractString
        s = strip(String(v))
        isempty(s) && return default
        parsed_i = tryparse(Int, s)
        if !isnothing(parsed_i)
            return parsed_i
        end
        parsed_f = tryparse(Float64, s)
        return isnothing(parsed_f) ? default : Int(round(parsed_f))
    end

    try
        return Int(v)
    catch
        return default
    end
end

"""
    load_timeseries(path::AbstractString; kwargs...)

Load a timeseries CSV file and normalize its schema to match the expected format.

This function reads a CSV file using CSV.jl and applies schema normalization
to ensure all required columns are present with appropriate data types and
default values.

# Arguments
- `path::AbstractString`: Path to the CSV file to load
- `kwargs...`: Additional keyword arguments passed to `CSV.read`

# Returns
- `DataFrame`: A DataFrame with normalized schema containing all required columns

# Examples
```julia
# Load a timeseries file
df = load_timeseries("data/experiment.csv")

# Load with specific CSV options
df = load_timeseries("data/experiment.csv", delim='\t', dateformat="yyyy-mm-dd")
```
"""
function load_timeseries(path::AbstractString; kwargs...)
    df = CSV.read(path, DataFrame; kwargs...)
    return normalize_schema(df)
end

"""
    normalize_schema(df::DataFrame; column_map=nothing, defaults=nothing)

Normalize a DataFrame to match the expected timeseries schema.

This function ensures that a DataFrame contains all required columns with
appropriate data types and default values. It can also remap column names
and fill missing values with defaults.

# Arguments
- `df::DataFrame`: The DataFrame to normalize
- `column_map::Dict{Symbol,Symbol}=Dict{Symbol,Symbol}()`: A dictionary mapping 
  source column names to target column names
- `defaults::Dict{Symbol,Any}=Dict{Symbol,Any}()`: Default values for missing columns

# Returns
- `DataFrame`: A normalized DataFrame with all required columns present

# Examples
```julia
# Normalize a DataFrame with default settings
normalized_df = normalize_schema(raw_df)

# Normalize with column mapping
normalized_df = normalize_schema(raw_df, column_map=Dict(:obs => :count))

# Normalize with custom defaults
normalized_df = normalize_schema(raw_df, defaults=Dict(:unit_time => "min"))
```
"""
    work = copy(df)
    current_names = Set(Symbol.(names(work)))
    had_dose = :dose in current_names
    had_treatment = :treatment_amount in current_names
    extra_columns = [name for name in Symbol.(names(work)) if !(name in REQUIRED_COLUMNS)]

    for (from, to) in column_map
        if from in current_names
            rename!(work, from => to)
            delete!(current_names, from)
            push!(current_names, to)
        end
    end

    # Auto-alias overlay-style columns: if :count is absent but :observed is present,
    # treat :observed as :count so that result/overlay CSVs load without error.
    if !(:count in current_names) && :observed in current_names
        rename!(work, :observed => :count)
        delete!(current_names, :observed)
        push!(current_names, :count)
        filter!(!=(Symbol(:observed)), extra_columns)
    end

    for col in REQUIRED_COLUMNS
        if !(col in current_names)
            default_value = get(defaults, col, missing)
            work[!, col] = fill(default_value, nrow(work))
            push!(current_names, col)
        end
    end

    ordered_columns = vcat(REQUIRED_COLUMNS, extra_columns)
    work = work[:, ordered_columns]

    time_vals = [_as_float_or_missing(v) for v in work.time]
    count_vals = [_as_float_or_missing(v) for v in work.count]
    valid_rows = map((t, c) -> !ismissing(t) && !ismissing(c), time_vals, count_vals)

    if any(.!valid_rows)
        work = work[valid_rows, :]
        time_vals = time_vals[valid_rows]
        count_vals = count_vals[valid_rows]
    end

    nrow(work) > 0 || error("No valid rows remain after dropping missing/non-numeric time or count values")

    work.time = Float64.(time_vals)
    work.count = Float64.(count_vals)

    if all(ismissing, work.error)
        work.error = fill(1.0, nrow(work))
    else
        parsed_error = [_as_float_or_missing(v) for v in work.error]
        work.error = [ismissing(v) ? 1.0 : Float64(v) for v in parsed_error]
    end

    if :treatment_amount in Symbol.(names(work))
        parsed_treatment = [_as_float_or_missing(v) for v in work.treatment_amount]
        work.treatment_amount = [ismissing(v) ? missing : Float64(v) for v in parsed_treatment]
    else
        work.treatment_amount = fill(missing, nrow(work))
    end

    dose_raw = [_as_float_or_missing(v) for v in work.dose]
    treatment_raw = [_as_float_or_missing(v) for v in work.treatment_amount]

    # Keep :dose and :treatment_amount synchronized to support either naming convention.
    if !had_dose && had_treatment
        work.dose = [ismissing(v) ? 0.0 : v for v in treatment_raw]
    else
        work.dose = [ismissing(v) ? 0.0 : v for v in dose_raw]
    end

    if had_dose && !had_treatment
        work.treatment_amount = copy(work.dose)
    else
        work.treatment_amount = [ismissing(treatment_raw[i]) ? work.dose[i] : treatment_raw[i] for i in eachindex(treatment_raw)]
    end

    work.replicate = [_as_int_or_default(v, 1) for v in work.replicate]

    return work
end

"""
    validate_timeseries(df::DataFrame)

Validate that a DataFrame conforms to the required timeseries schema.

This function checks that the DataFrame contains all required columns,
that time and count values are valid (finite, nonnegative for count),
that time is monotonic within each experimental condition, and that
error values are positive.

# Arguments
- `df::DataFrame`: The DataFrame to validate

# Returns
- `Bool`: true if validation passes

# Examples
```julia
# Validate a timeseries DataFrame
is_valid = validate_timeseries(df)

# This will throw an error if validation fails
validate_timeseries(df)
```
"""
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

"""
    validate_required_metadata(
        df::DataFrame;
        required_metadata::Vector{Symbol} = copy(STRICT_REQUIRED_METADATA),
    )

Validate that a DataFrame contains all required metadata columns with valid values.

This function checks for the presence of required metadata columns and validates
that they contain meaningful values (not entirely missing, not containing only
missing values, and for string columns, not containing empty strings).

# Arguments
- `df::DataFrame`: The DataFrame to validate
- `required_metadata::Vector{Symbol}=copy(STRICT_REQUIRED_METADATA)`: 
  The metadata columns to validate for presence and validity

# Returns
- `Bool`: true if validation passes

# Examples
```julia
# Validate required metadata using defaults
is_valid = validate_required_metadata(df)

# Validate with custom required metadata
is_valid = validate_required_metadata(df, required_metadata=[:cell_line, :density])
```
"""
function validate_required_metadata(
    df::DataFrame;
    required_metadata::Vector{Symbol} = copy(STRICT_REQUIRED_METADATA),
)
    cols = Symbol.(names(df))
    missing_cols = setdiff(required_metadata, cols)
    if !isempty(missing_cols)
        error("Strict schema mode: missing required metadata columns: $(join(string.(missing_cols), ", "))")
    end

    issues = String[]
    for col in required_metadata
        values = df[!, col]
        if all(ismissing, values)
            push!(issues, string(col) * " is entirely missing")
            continue
        end

        if any(ismissing, values)
            push!(issues, string(col) * " contains missing values")
        end

        if eltype(values) <: AbstractString || any(v -> v isa AbstractString, values)
            if any(v -> !ismissing(v) && isempty(strip(String(v))), values)
                push!(issues, string(col) * " contains empty-string values")
            end
        end
    end

    if !isempty(issues)
        error("Strict schema mode: metadata validation failed: " * join(issues, "; "))
    end

    return true
end

end # module DataLayer
