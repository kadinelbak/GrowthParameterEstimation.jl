using V1SimpleODE
using Test
using CSV, DataFrames, DifferentialEquations

# Get the directory where the test file is located
const TEST_DIR = dirname(@__FILE__)

@testset "Basic V1SimpleODE Tests" begin
    println("Starting basic tests...")
    
    # Test 1: Module loading
    @test V1SimpleODE isa Module
    println("✓ Module loaded successfully")
    
    # Test 2: Create test data
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [10.0, 20.0, 35.0, 55.0, 80.0]
    @test length(x) == length(y)
    @test length(x) > 0
    println("✓ Test data created: $(length(x)) points")
    
    # Test 3: Built-in models
    u_test = [100.0]
    du_test = similar(u_test)
    
    # Test logistic model
    V1SimpleODE.logistic_growth!(du_test, u_test, [0.1, 500.0], 1.0)
    @test isfinite(du_test[1])
    println("✓ Logistic model works")
    
    # Test Gompertz model
    V1SimpleODE.gompertz_growth!(du_test, u_test, [0.05, 2.0, 500.0], 1.0)
    @test isfinite(du_test[1])
    println("✓ Gompertz model works")
    
    println("All basic tests passed!")
end
