# Feature Specification: Enhanced Function Tour with Complete Staged Pipeline Example

## Overview
This feature spec outlines a plan to enhance the existing `function_tour.ipynb` notebook to include a complete staged pipeline example that demonstrates the package's capability to handle inherited parameter workflows. The example will start with monoculture untreated data and progress to treated coculture data, showcasing how parameters from earlier stages can be used to inform later stages.

## Current State Assessment
The existing `function_tour.ipynb` notebook provides:
- Comprehensive API walkthrough covering data layer, exposure profiles, model registry/simulation, fitting, analysis, and basic pipeline workflow
- A synthetic joint-fitting example
- However, it lacks a complete staged pipeline example demonstrating inherited parameter workflows

## Goals
1. Enhance the function tour notebook with a complete staged pipeline example
2. Demonstrate the workflow from monoculture untreated to treated coculture data
3. Showcase parameter inheritance between stages
4. Use generated data that follows commonly known biological growth models
5. Provide clear visualization and interpretation of results at each stage

## Scope
### Included
- Modification of `tests/function_tour.ipynb` to add a staged pipeline section
- Generation of synthetic data representing:
  - Monoculture untreated (baseline growth)
  - Monoculture treated (drug response)
  - Coculture untreated (interaction effects)
  - Coculture treated (combined drug and interaction effects)
- Implementation of a staged pipeline with proper parameter inheritance
- Visualization of results at each stage
- Explanation of the biological rationale behind the models

### Excluded
- Changes to the core package API
- Modifications to other documentation or examples
- Removal of existing content from the function tour

## Approach

### 1. Synthetic Data Generation
Create biologically plausible synthetic datasets:
- **Monoculture Untreated**: Logistic growth with known parameters (r=0.5, K=1e6)
- **Monoculture Treated**: Logistic growth with death term (r=0.5, K=1e6, death_rate=0.3)
- **Coculture Untreated**: Two-state logistic growth with interaction (competition or cooperation)
- **Coculture Treated**: Two-state logistic growth with death term on one population and interaction

### 2. Staged Pipeline Implementation
Create a pipeline with 4 stages:
1. **Stage 1**: Fit monoculture untreated to establish baseline growth parameters
2. **Stage 2**: Fit monoculture treated, inheriting growth parameters from Stage 1, estimating death rate
3. **Stage 3**: Fit coculture untreated, inheriting growth parameters, estimating interaction parameters
4. **Stage 4**: Fit coculture treated, inheriting all previous parameters, estimating combined effects

### 3. Notebook Enhancements
Add a new section to the function tour notebook titled "Staged Pipeline Example: From Monoculture to Coculture" containing:
- Data generation code with comments explaining biological rationale
- Staged pipeline configuration with clear stage definitions
- Execution of the staged pipeline
- Visualization of results at each stage
- Comparison of estimated vs. true parameters
- Discussion of how parameter inheritance improves fitting

## Acceptance Criteria
- [ ] The enhanced function tour notebook includes a complete staged pipeline example
- [ ] The example demonstrates parameter inheritance across 4 stages
- [ ] The synthetic data follows recognizable biological growth models
- [ ] The notebook runs successfully from start to finish
- [ ] Visualizations are generated and displayed appropriately
- [ ] The example clearly shows the value of the staged pipeline approach
- [ ] All code is well-commented and educational

## Implementation Steps

### Phase 1: Data Generation
1. [ ] Create synthetic time series data for monoculture untreated
2. [ ] Create synthetic time series data for monoculture treated
3. [ ] Create synthetic time series data for coculture untreated
4. [ ] Create synthetic time series data for coculture treated
5. [ ] Add realistic noise to all datasets

### Phase 2: Staged Pipeline Configuration
1. [ ] Define Stage 1: Monoculture untreated fitting
2. [ ] Define Stage 2: Monoculture treated fitting with parameter inheritance
3. [ ] Define Stage 3: Coculture untreated fitting with parameter inheritance
4. [ ] Define Stage 4: Coculture treated fitting with parameter inheritance
5. [ ] Create PipelineConfig object

### Phase 3: Notebook Integration
1. [ ] Add new markdown section to function_tour.ipynb
2. [ ] Add data generation code with explanations
3. [ ] Add staged pipeline setup and execution code
4. [ ] Add visualization code for each stage
5. [ ] Add results comparison and interpretation
6. [ ] Ensure all cells execute correctly

### Phase 4: Verification
1. [ ] Run the enhanced notebook completely to verify functionality
2. [ ] Check that all visualizations are generated
3. [ ] Verify that parameter inheritance works as expected
4. [ ] Confirm the educational value of the example

## References
- Existing function_tour.ipynb notebook
- Staged pipeline documentation in README.md
- Biological growth models: logistic, Gompertz, and interaction models
- Parameter inheritance patterns in staged workflows
