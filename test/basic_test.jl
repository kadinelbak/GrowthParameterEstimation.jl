using V1SimpleODE
using Test
using CSV, DataFrames

# Get the directory where the test file is located
const TEST_DIR = dirname(@__FILE__)

@testset "Basic V1SimpleODE Tests" begin
    # Test data loading
    df = CSV.read(joinpath(TEST_DIR, "test_data.csv"), DataFrame)
    @test names(df) == ["Day Averages"]
    
    # Test data extraction
    x, y = V1SimpleODE.extractData(df)
    @test length(x) == length(y)
    @test length(x) > 0
    @test all(x .== 1:length(x))
    
    println("Basic tests passed!")
end
