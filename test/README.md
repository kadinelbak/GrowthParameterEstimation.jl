# V1SimpleODE Test Suite

This directory contains a comprehensive test suite for the V1SimpleODE package.

## Test Structure

### Files:

1. **`runtests.jl`** - Main test runner that executes all tests
2. **`setup_test_env.jl`** - Script to install required testing dependencies
3. **`test_data_generator.jl`** - Functions to generate synthetic test data
4. **`test_fitting.jl`** - Tests for fitting functionality
5. **`test_analysis.jl`** - Tests for analysis functionality  
6. **`simple_test.jl`** - Simple standalone test for development

### Test Data Generation

The `test_data_generator.jl` file provides functions to create synthetic logistic growth data:

- `generate_logistic_test_data()` - Creates single dataset with noise
- `generate_three_test_datasets()` - Creates three different datasets for testing
- `get_basic_test_data()` - Simple dataset for basic tests

### Test Coverage

#### Fitting Tests (`test_fitting.jl`):
- Basic single fit functionality
- Different ODE models (logistic, Gompertz, exponential variants)
- Model comparison capabilities
- Three datasets fitting
- Parameter bounds handling
- Fixed parameters functionality
- Error handling

#### Analysis Tests (`test_analysis.jl`):
- Leave-one-out cross-validation
- K-fold cross-validation  
- Parameter sensitivity analysis
- Residual analysis and diagnostics
- Enhanced BIC model comparison
- Integration tests across different models
- Error handling in analysis functions

## Running Tests

### Method 1: Standard Julia Testing
```julia
using Pkg
Pkg.test()
```

### Method 2: Direct Execution
```julia
include("runtests.jl")
```

### Method 3: Simple Development Test
```julia
include("simple_test.jl")
```

### Method 4: Individual Test Files
```julia
include("test_fitting.jl")    # Run only fitting tests
include("test_analysis.jl")   # Run only analysis tests
```

## Test Data Characteristics

The synthetic test data emulates realistic growth scenarios:

- **Dataset 1**: Fast growth (r=0.25), moderate capacity (K=80)
- **Dataset 2**: Slow growth (r=0.15), high capacity (K=150)  
- **Dataset 3**: Medium growth (r=0.18), low capacity (K=60)

All datasets include realistic noise levels (3-6%) and use different time ranges.

## Expected Outputs

When tests pass successfully, you should see:
- ✓ markers for passed tests
- Performance metrics (RMSE, R², BIC values)
- Parameter estimation results
- Validation statistics

## Dependencies

Required packages (installed by `setup_test_env.jl`):
- Test
- DifferentialEquations
- StatsBase
- Plots
- BlackBoxOptim
- Statistics
- Random
- Distributions

## Notes

- Tests use fixed random seeds for reproducibility
- All plotting is disabled in tests for speed
- Tests handle edge cases and error conditions
- Analysis functions are tested with multiple ODE models
