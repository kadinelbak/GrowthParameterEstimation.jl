# Feature Specification: Fix Function Tour Staged Pipeline Example

## Overview
This feature spec outlines a plan to fix the existing `function_tour.ipynb` notebook's staged pipeline example so that it runs successfully without errors. The current implementation fails because it references models that are not registered in the package's model registry.

## Current State Assessment
The existing `function_tour.ipynb` notebook contains a staged pipeline section (lines 609-963) that attempts to demonstrate parameter inheritance across four stages:
1. Monoculture untreated (baseline growth)
2. Monoculture treated (adding drug effect)
3. Coculture untreated (adding population interaction)
4. Coculture treated (combining drug and interaction effects)

However, this section fails to execute because:
- It references `"logistic_growth_with_death"` which is not registered in the model registry
- It references interaction models that are not properly defined for the use case
- The notebook includes a catch block that shows mock results instead of actual pipeline execution

## Goals
1. Fix the function tour notebook's staged pipeline example to run successfully
2. Register missing models that are needed for the staged pipeline example
3. Ensure the example properly demonstrates parameter inheritance across stages
4. Remove the mock result fallback and show actual pipeline execution

## Scope
### Included
- Registration of `LogisticWithDeathModel` and `GompertzWithDeathModel` in the model registry
- Modification of `tests/notebooks/function_tour.ipynb` to use correct model names
- Verification that the staged pipeline example executes successfully from start to finish

### Excluded
- Changes to the core package API beyond model registration
- Modifications to other documentation or examples
- Removal of existing content from the function tour (except fixing broken references)

## Approach

### 1. Register Missing Models
Add registration for the missing death models in `src/registry.jl`:
- `LogisticWithDeathModel` 
- `GompertzWithDeathModel`

### 2. Update Function Tour Notebook
Modify `tests/notebooks/function_tour.ipynb` to:
- Replace `"logistic_growth_with_death"` with the correct registered model name
- Ensure all referenced models exist in the registry
- Remove the catch block that shows mock results (since the pipeline should now work)
- Keep the educational content about parameter inheritance

### 3. Verification
Ensure the enhanced notebook runs successfully from start to finish, demonstrating:
- Proper data generation for all four stages
- Successful execution of the staged pipeline with parameter inheritance
- Visualization of results at each stage
- Comparison of estimated vs. true parameters

## Acceptance Criteria
- [ ] The function tour notebook's staged pipeline example runs successfully without errors
- [ ] The example demonstrates parameter inheritance across 4 stages using real pipeline execution
- [ ] All models referenced in the notebook are properly registered in the model registry
- [ ] Visualizations are generated and displayed appropriately
- [ ] The example clearly shows the value of the staged pipeline approach
- [ ] No mock result fallbacks are needed - the pipeline executes completely

## Implementation Steps

### Phase 1: Model Registration
1. [ ] Add registration for `LogisticWithDeathModel` in `src/registry.jl`
2. [ ] Add registration for `GompertzWithDeathModel` in `src/registry.jl`

### Phase 2: Notebook Fixes
1. [ ] Update `tests/notebooks/function_tour.ipynb` to use correct model names
2. [ ] Remove the catch block that shows mock results
3. [ ] Ensure all cells in the staged pipeline section execute correctly

### Phase 3: Verification
1. [ ] Run the enhanced notebook completely to verify functionality
2. [ ] Check that all visualizations are generated
3. [ ] Verify that parameter inheritance works as expected
4. [ ] Confirm the educational value of the example

## References
- Existing function_tour.ipynb notebook
- Model definitions in src/models.jl
- Model registration in src/registry.jl
- Biological growth models: logistic, Gompertz