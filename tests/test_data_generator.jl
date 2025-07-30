# Test data generation for GrowthParamEst tests
# Creates synthetic logistic growth data with realistic noise

using Random
using DifferentialEquations
using Statistics

"""
    generate_logistic_test_data(; 
        n_points::Int = 10,
        noise_level::Float64 = 0.05,
        r::Float64 = 0.2,
        K::Float64 = 100.0,
        y0::Float64 = 5.0,
        t_end::Float64 = 20.0,
        seed::Int = 42
    )

Generate synthetic logistic growth data with noise for testing purposes.

Returns a named tuple with x (time points) and y (noisy observations).
"""
function generate_logistic_test_data(; 
    n_points::Int = 10,
    noise_level::Float64 = 0.05,
    r::Float64 = 0.2,
    K::Float64 = 100.0,
    y0::Float64 = 5.0,
    t_end::Float64 = 20.0,
    seed::Int = 42
)
    Random.seed!(seed)
    
    # Time points
    x = range(0.0, t_end, length=n_points)
    
    # True logistic solution
    function logistic_solution(t, r, K, y0)
        return K / (1 + ((K - y0) / y0) * exp(-r * t))
    end
    
    # Generate true values
    y_true = [logistic_solution(t, r, K, y0) for t in x]
    
    # Add noise
    noise = randn(n_points) .* (noise_level * maximum(y_true))
    y = y_true .+ noise
    
    # Ensure positive values
    y = max.(y, 0.1)
    
    return (x = collect(x), y = y, y_true = y_true, params_true = [r, K])
end

"""
    generate_three_test_datasets()

Generate three different logistic growth datasets for testing fit_three_datasets function.
Each dataset has different parameters but follows logistic growth pattern.
"""
function generate_three_test_datasets()
    # Dataset 1: Fast growth, moderate capacity
    data1 = generate_logistic_test_data(
        n_points = 12,
        noise_level = 0.03,
        r = 0.25,
        K = 80.0,
        y0 = 3.0,
        t_end = 15.0,
        seed = 41
    )
    
    # Dataset 2: Slow growth, high capacity  
    data2 = generate_logistic_test_data(
        n_points = 15,
        noise_level = 0.04,
        r = 0.15,
        K = 150.0,
        y0 = 8.0,
        t_end = 25.0,
        seed = 42
    )
    
    # Dataset 3: Medium growth, low capacity
    data3 = generate_logistic_test_data(
        n_points = 10,
        noise_level = 0.06,
        r = 0.18,
        K = 60.0,
        y0 = 5.0,
        t_end = 18.0,
        seed = 43
    )
    
    return (data1, data2, data3)
end

"""
    get_basic_test_data()

Get a simple dataset for basic functionality tests.
"""
function get_basic_test_data()
    return generate_logistic_test_data(
        n_points = 8,
        noise_level = 0.02,
        r = 0.2,
        K = 50.0,
        y0 = 2.0,
        t_end = 12.0,
        seed = 100
    )
end
