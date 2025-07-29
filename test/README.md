# V1SimpleODE Test Suite

This folder contains organized tests for the V1SimpleODE package.

## Test Structure

### Main Test Files

- **`runtests.jl`** - Main test runner that executes all tests
- **`individual_tests.jl`** - Individual test functions that can be run separately
- **`test_fitting.jl`** - Focused tests for the Fitting module
- **`test_analysis.jl`** - Focused tests for the Analysis module
- **`test_data_generator.jl`** - Utility functions for generating test data

### Demo and Example Files

- **`simple_test.jl`** - Quick development test script
- **`working_demo.jl`** - Comprehensive demonstration of package features
- **`demo_script.jl`** - Additional demo examples

### Utility Files

- **`setup_test_env.jl`** - Test environment setup
- **`fix_tests.jl`** - Test utilities and fixes

## Running Tests

### Run All Tests
```julia
julia --project=. test/runtests.jl
```

### Run Individual Test Functions
```julia
# Load the individual test functions
include("test/individual_tests.jl")

# Run specific tests
test_package_loading()
test_basic_fitting()
test_cross_validation()
test_sensitivity_analysis()

# Or run all individual tests
run_all_individual_tests()
```

### Run Module-Specific Tests
```julia
# Run only fitting tests
include("test/test_fitting.jl")

# Run only analysis tests
include("test/test_analysis.jl")
```

## Individual Test Functions

The `individual_tests.jl` file provides these functions for focused testing:

- `test_package_loading()` - Verify package loads correctly
- `test_ode_models()` - Test all ODE model functions
- `test_basic_fitting()` - Test basic fitting functionality
- `test_model_comparison()` - Test model comparison features
- `test_cross_validation()` - Test cross-validation methods
- `test_sensitivity_analysis()` - Test parameter sensitivity analysis
- `test_residual_analysis()` - Test residual analysis
- `test_enhanced_bic_analysis()` - Test enhanced BIC model selection
- `test_three_datasets()` - Test multi-dataset fitting

## Test Data

Test data is generated using functions in `test_data_generator.jl`:

- `get_basic_test_data()` - Standard logistic growth test data
- `generate_logistic_test_data(...)` - Customizable synthetic data
- `generate_three_test_datasets()` - Multiple datasets for comparison

## Quick Development Testing

For quick testing during development, use:

```julia
# Quick functional test
include("test/simple_test.jl")

# Comprehensive demo
include("test/working_demo.jl")
```

## Test Organization Benefits

1. **Individual Testing**: Each function can be tested in isolation
2. **Module Separation**: Fitting and Analysis tests are separate
3. **No Duplication**: Removed redundant test code
4. **Clear Structure**: Easy to find and run specific tests
5. **Development Friendly**: Quick testing during development
