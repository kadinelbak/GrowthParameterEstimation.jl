# Test Organization and Output Setup

## Goal
- Fix formatting issues in test notebooks (function_tour.ipynb, pipeline_assessment.ipynb, pipeline_step_by_step_template.ipynb)
- Ensure all test notebooks direct their outputs to a single `outputs` folder within the `tests` directory
- Organize test files by moving notebooks to `tests/notebooks/` and Julia scripts to `tests/julia/`

## Details

### Formatting Fix
The notebooks contain parse errors due to improper escaping of double quotes inside Julia strings for `output_csv` parameters.
Specifically, lines like:
```julia
output_csv=joinpath(@__DIR__, "..", "outputs", "function_tour", "nb_compare.csv")
```
are causing `ParseError: cannot juxtapose string literal` when the notebook is executed because of the way the JSON notebook format handles escaped quotes.

We need to fix these lines by either:
1. Using single quotes for the inner string: `output_csv=joinpath(@__DIR__, "..", "outputs", "function_tour", 'nb_compare.csv')`
2. Or defining the output directory first and then using it: 
   ```julia
   output_dir = joinpath(@__DIR__, "..", "outputs", "function_tour")
   output_csv=joinpath(output_dir, "nb_compare.csv")
   ```

### Output Directory Setup
After moving the notebooks to `tests/notebooks/`, the current `joinpath(@__DIR__, "..", "outputs", ...)` will correctly point to `tests/outputs/` (since `__DIR__` is `tests/notebooks/` and `..` goes to `tests/`).

We want each notebook's output to go into a subdirectory named after the notebook:
- `function_tour.ipynb` → `tests/outputs/function_tour/`
- `pipeline_assessment.ipynb` → `tests/outputs/pipeline_assessment/`
- `pipeline_step_by_step_template.ipynb` → `tests/outputs/pipeline_step_by_step/`

### File Organization
Move all `.ipynb` files from the `test/` directory to `tests/notebooks/`
Move all `.jl` files from the `test/` directory to `tests/julia/`
Ensure the `tests/outputs/` directory exists.

## Steps to Implement

1. Create `tests/notebooks/` and `tests/julia/` directories if they don't exist
2. Move all `.ipynb` files from `test/` to `tests/notebooks/`
3. Move all `.jl` files from `test/` to `tests/julia/`
4. Create `tests/outputs/` directory
5. For each notebook in `tests/notebooks/`:
   - Fix the formatting issue in the `output_csv` lines
   - Ensure the output directory is set to `tests/outputs/<notebook_name_without_extension>/`
6. Execute each notebook to verify it runs without errors and produces output in the correct location
7. Execute the Julia test scripts in `tests/julia/` to ensure they still work

## Expected Outcome
- All three notebooks execute successfully without parse errors
- Output files are generated in `tests/outputs/<notebook_name>/` for each notebook
- Test files are organized: notebooks in `tests/notebooks/`, Julia scripts in `tests/julia/`
- The `tests/outputs/` directory contains subdirectories for each notebook with their respective output files
