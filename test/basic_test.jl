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
    
    # Test 2: Data loading
    df = CSV.read(joinpath(TEST_DIR, "test_data.csv"), DataFrame)
    @test names(df) == ["Day Averages"]
    println("✓ Test data loaded")
    
    # Test 3: Data extraction
    x, y = V1SimpleODE.extractData(df)
    @test length(x) == length(y)
    @test length(x) > 0
    @test all(x .== 1:length(x))
    println("✓ Data extraction works: extracted $(length(x)) points")
    
    # Test 4: Built-in models
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
