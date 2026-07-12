# Feature Specification: Organize Tests and Fix Notebook Formatting Issues

## Overview
This feature spec outlines tasks to:
1. Fix formatting issues in Jupyter notebooks that prevent execution (specifically escaped dollar signs in LaTeX).
2. Ensure all test notebooks (function_tour.ipynb, pipeline_assessment.ipynb, pipeline_step_by_step_template.ipynb) direct their outputs to a single `outputs` folder within the `test` directory.
3. Organize test files by moving Julia scripts (.jl) to a `test/julia/` subdirectory and notebooks (.ipynb) to a `test/notebooks/` subdirectory.
4. Update any necessary internal references to reflect the new file locations.

## Current State Assessment
- The `test/function_tour.ipynb` contains a parse error due to unescaped `$` in a LaTeX table within a code cell's output (or source?), causing `ParseError: identifier or parenthesized expression expected after $ in string`.
- The `test/pipeline_assessment.ipynb` and `test/pipeline_step_by_step_template.ipynb` may have similar issues or not; they should be checked.
- Notebooks currently likely write outputs to various locations (e.g., temporary directories) without consolidation.
- The `test/` directory contains a mix of `.jl` and `.ipynb` files, making organization less clear.

## Goals
1. Fix the parsing error in `function_tour.ipynb` by properly escaping or removing the problematic `$` in the LaTeX string.
2. Ensure all three notebooks write any generated files (plots, data, etc.) to a centralized `test/outputs/` directory.
3. Move all `.jl` files from `test/` into `test/julia/`.
4. Move all `.ipynb` files from `test/` into `test/notebooks/`.
5. Verify that after reorganization, the notebooks still execute correctly and can find any required dependencies (e.g., they may need to adjust relative paths to the project root).

## Scope
### Included
- Editing `test/function_tour.ipynb` to fix the LaTeX formatting.
- Modifying the configuration cells in all three notebooks to set an output directory to `test/outputs/` (relative to the notebook location).
- Creating directories `test/julia/` and `test/notebooks/` and moving files accordingly.
- Updating any notebook cells that reference files via relative paths if needed (though notebooks likely use `pwd()` or `@__DIR__`).
- Ensuring the `test/outputs/` directory exists.

### Excluded
- Modifications to the source code in `src/`.
- Changes to documentation outside of the test notebooks.
- Altering the core functionality of the package.

## Approach

### 1. Fix Notebook Formatting
- Locate the problematic cell in `function_tour.ipynb` (around line 387) where a string contains `$\\dots$`.
- Replace the `$` with `\\$` to escape it for Julia string interpolation, or use a raw string if appropriate. Since the string is within a code cell that is likely meant to be printed as LaTeX, we need to ensure the literal `$` appears in the output. In Julia string, to get a literal `$` we write `\\$`. So change `$\dots$` to `\\$\\dots\\$`.
- Alternatively, if the string is intended for display in a Markdown cell, but it's in a code cell, we need to see context. We'll examine and apply the fix.

### 2. Consolidate Output Directory
- In each notebook, find where the pipeline configuration is set (likely a call to `default_config` or similar) and modify the `output_dir` argument to point to `joinpath(@__DIR__, "..", "outputs")` or `joinpath(pwd(), "outputs")` depending on the notebook's working directory.
- Ensure the directory exists by adding a `mkpath` call if necessary.

### 3. Organize Test Files
- Create `test/julia/` and `test/notebooks/` directories.
- Move all `.jl` files from `test/` to `test/julia/`.
- Move all `.ipynb` files from `test/` to `test/notebooks/`.
- After moving, verify that notebooks still work (they likely use `@__DIR__` to locate the project root, which should still be valid because moving notebooks into a subdirectory changes `@__DIR__`; we may need to adjust path calculations to go up two levels instead of one). We'll check and adjust if necessary.

### 4. Verification
- Run each notebook to ensure they execute without errors and produce outputs in `test/outputs/`.
- Run the Julia test files to ensure they still work.

## Acceptance Criteria
- [ ] `function_tour.ipynb` executes without parse errors.
- [ ] `pipeline_assessment.ipynb` and `pipeline_step_by_step_template.ipynb` execute without errors.
- [ ] All notebooks write outputs to `test/outputs/` (or subdirectories thereof).
- [ ] All `.jl` files are located in `test/julia/`.
- [ ] All `.ipynb` files are located in `test/notebooks/`.
- [ ] No broken references due to reorganization.

## Implementation Steps

### Phase 1: Fix Formatting
1. Backup `test/function_tour.ipynb`.
2. Edit the problematic cell to escape the dollar signs.
3. Save the file.

### Phase 2: Set Output Directory
1. For each notebook, locate the configuration block (look for `default_config` or `PipelineConfig`).
2. Modify the `output_dir` parameter to point to a centralized outputs folder.
3. Ensure the folder is created if it doesn't exist.

### Phase 3: Reorganize Files
1. Create `test/julia/` and `test/notebooks/`.
2. Move files accordingly.
3. Adjust any path references in notebooks that depend on being in the test root.

### Phase 4: Verification
1. Execute each notebook and confirm they run to completion.
2. Check that outputs appear in `test/outputs/`.
3. Run the Julia test scripts from their new location to ensure they still work.

## References
- Existing notebooks in `test/`.
- File organization plan from `FILE_ORGANIZATION_PLAN.md`.