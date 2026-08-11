# Data layer module - Contains data loading, normalization, and validation functions
module DataLayer

using CSV
using DataFrames
using Dates

export
    REQUIRED_COLUMNS,
    normalize_schema,
    validate_timeseries,
    validate_required_metadata,
    load_timeseries,
    STRICT_REQUIRED_METADATA

"""
    STRICT_REQUIRED_METADATA

Metadata fields that are strictly required for processing.
These must be present in the normalized schema for strict validation to pass.
"""
const STRICT_REQUIRED_METADATA = [:time, :count, :error, :dose, :cell_line, :density, :replicate]
const REQUIRED_COLUMNS = [:time, :count, :error, :dose, :treatment_amount, :cell_line, :density, :replicate]

"""
    normalize_schema(df::DataFrame) -> DataFrame

Normalize a raw data DataFrame to the expected schema.
Converts column names to standard names and ensures proper data types.

# Arguments
- `df::DataFrame`: Raw data DataFrame with experimental measurements

# Returns
- `DataFrame`: Normalized DataFrame with standard column names and types

# Notes
Expected columns:
- `time`: Measurement time points (Float64)
- `count`: Observed measurements/counts (Float64)
- `error`: Measurement uncertainties/errors (Float64)
- `dose`: Drug/exposure concentration (Float64)
- `cell_line`: Cell line identifier (String)
- `density`: Initial cell density (Float64)
- `replicate`: Replicate number (Int)
"""
function normalize_schema(df::DataFrame)
    # Create a copy to avoid modifying the original
    normalized = copy(df)
    
    # Standardize column names (case-insensitive mapping)
    col_mapping = Dict(
        "time" => :time,
        "Time" => :time,
        "TIME" => :time,
        "count" => :count,
        "Count" => :count,
        "COUNT" => :count,
        "error" => :error,
        "Error" => :error,
        "ERROR" => :error,
        "uncertainty" => :error,
        "Uncertainty" => :error,
        "UNCERTAINTY" => :error,
        "dose" => :dose,
        "Dose" => :dose,
        "DOSE" => :dose,
        "concentration" => :dose,
        "Concentration" => :dose,
        "CONCENTRATION" => :dose,
        "TreatmentAmount" => :treatment_amount,
        "treatment_amount" => :treatment_amount,
        "TREATMENT_AMOUNT" => :treatment_amount,
        "cell_line" => :cell_line,
        "CellLine" => :cell_line,
        "CELL_LINE" => :cell_line,
        "cellline" => :cell_line,
        "Cellline" => :cell_line,
        "density" => :density,
        "Density" => :density,
        "DENSITY" => :density,
        "initial_density" => :density,
        "InitialDensity" => :density,
        "replicate" => :replicate,
        "Replicate" => :replicate,
        "REPLICATE" => :replicate,
        "rep" => :replicate,
        "Rep" => :replicate,
    )
    
    # Rename columns according to mapping
    for (old_name, new_name) in col_mapping
        if old_name in names(normalized)
            rename!(normalized, old_name => new_name)
        end
    end
    
    if :dose in Symbol.(names(normalized)) && !(:treatment_amount in Symbol.(names(normalized)))
        normalized[!, :treatment_amount] = copy(normalized[!, :dose])
    elseif :treatment_amount in Symbol.(names(normalized)) && !(:dose in Symbol.(names(normalized)))
        normalized[!, :dose] = copy(normalized[!, :treatment_amount])
    end

    # Ensure required columns exist, add missing ones with defaults
    required_cols = REQUIRED_COLUMNS
    for col in required_cols
        if !(col in Symbol.(names(normalized)))
            if col == :time || col == :count || col == :error || col == :dose || col == :treatment_amount || col == :density
                normalized[!, col] = fill(0.0, nrow(normalized))
            elseif col == :cell_line
                normalized[!, col] = fill("unknown", nrow(normalized))
            elseif col == :replicate
                normalized[!, col] = fill(1, nrow(normalized))
            end
        end
    end
    
    # Convert data types
    normalized.time = Float64.(normalized.time)
    normalized.count = Float64.(normalized.count)
    normalized.error = Float64.(normalized.error)
    normalized.dose = Float64.(normalized.dose)
    normalized.treatment_amount = Float64.(normalized.treatment_amount)
    normalized.cell_line = String.(normalized.cell_line)
    normalized.density = Float64.(normalized.density)
    normalized.replicate = Int.(normalized.replicate)
    
    # Ensure errors are positive (replace zeros with small positive value)
    normalized.error = max.(normalized.error, 1e-12)
    
    return normalized
end

"""
    validate_timeseries(df::DataFrame) -> Bool

Validate that a normalized DataFrame contains valid time series data.
Checks for monotonic time, non-negative counts, and positive errors.

# Arguments
- `df::DataFrame`: Normalized data DataFrame

# Returns
- `Bool`: True if validation passes

# Throws
- `ErrorException` if validation fails
"""
function validate_timeseries(df::DataFrame)
    # Check for monotonic time within each observed trajectory/condition.
    grouping_cols = [col for col in [:cell_line, :density, :dose, :replicate] if col in Symbol.(names(df))]
    groups = isempty(grouping_cols) ? [df] : groupby(df, grouping_cols)
    for group in groups
        if !all(diff(group.time) .>= 0)
            throw(ErrorException("Time values are not monotonic increasing within a condition"))
        end
    end
    
    # Check for non-negative counts
    if any(df.count .< 0)
        throw(ErrorException("Count values must be non-negative"))
    end
    
    # Check for positive errors
    if any(df.error .<= 0)
        throw(ErrorException("Error values must be positive"))
    end
    
    return true
end

"""
    validate_required_metadata(df::DataFrame; required_metadata::Vector{Symbol}=STRICT_REQUIRED_METADATA) -> Bool

Validate that all required metadata fields are present and have valid data.

# Arguments
- `df::DataFrame`: Normalized data DataFrame
- `required_metadata::Vector{Symbol}`: List of required metadata fields to check

# Returns
- `Bool`: True if all required metadata is present and valid

# Throws
- `ErrorException` if validation fails
"""
function validate_required_metadata(df::DataFrame; required_metadata::Vector{Symbol}=STRICT_REQUIRED_METADATA)
    # Check that all required columns exist
    missing_cols = setdiff(required_metadata, Symbol.(names(df)))
    if !isempty(missing_cols)
        throw(ErrorException("Missing required metadata fields: $(string.(missing_cols))"))
    end
    
    # Check that required columns have non-missing data
    for col in required_metadata
        if any(ismissing, df[!, col])
            throw(ErrorException("Required metadata field $(col) contains missing values"))
        end
    end
    
    return true
end

"""
    load_timeseries(file_path::AbstractString) -> DataFrame

Load time series data from a CSV file and normalize the schema.

# Arguments
- `file_path::AbstractString`: Path to CSV file containing time series data

# Returns
- `DataFrame`: Normalized DataFrame with standardized schema

# Notes
Supported formats:
- CSV files with headers containing variations of expected column names
"""
function load_timeseries(file_path::AbstractString)
    if !isfile(file_path)
        throw(ArgumentError("File not found: $file_path"))
    end
    
    # Try to read as CSV
    raw_data = try
        CSV.File(file_path; ignorerepeated=false) |> DataFrame
    catch
        throw(ArgumentError("Unable to read file as CSV: $file_path"))
    end
    
    # Normalize the schema
    return normalize_schema(raw_data)
end

end # module DataLayer
