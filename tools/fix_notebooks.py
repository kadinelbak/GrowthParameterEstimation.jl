import json
import sys

notebooks = [
    'test/notebooks/function_tour.ipynb',
    'test/notebooks/pipeline_assessment.ipynb',
    'test/notebooks/pipeline_step_by_step_template.ipynb'
]

for nb_path in notebooks:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for i, line in enumerate(cell['source']):
                # Replace output_csv lines that have the old format
                if 'output_csv=joinpath(@__DIR__, "..", "outputs",' in line and '\\"nb_compare.csv\\"' in line:
                    # Replace with: output_csv=joinpath(output_dir, "nb_compare.csv")
                    # But we need to see the context: we want to have output_dir defined above.
                    # We'll do a more general replacement: replace the whole line if it matches the pattern.
                    # We'll assume that the line is exactly the one we want to replace.
                    # We'll split the line to get the notebook name.
                    # Instead, we'll do a simple replacement: change the joinpath to use output_dir if output_dir is defined in the same cell.
                    # Since we are going to restructure the cell, let's do a different approach: we will rebuild the cell for the specific lines.
                    pass  # We'll handle by reconstructing the cell for the known pattern.
    # Instead of per-line, let's reconstruct the known problematic cell for each notebook.
    # We know the pattern: we want to replace the lines after "# Fitting + comparisons" for function_tour, and similar for others.
    # Given the time, we'll do a simpler fix: just replace the problematic strings in the entire source string.
    # Join the source lines, do string replace, then split again.
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            full_source = ''.join(cell['source'])
            # Replace the problematic patterns
            full_source = full_source.replace('output_csv=joinpath(@__DIR__, "..", "outputs", "function_tour", "\\"nb_compare.csv\\"")', 'output_csv=joinpath(output_dir, "nb_compare.csv")')
            full_source = full_source.replace('output_csv=joinpath(@__DIR__, "..", "outputs", "function_tour", "\\"nb_compare_dict.csv\\"")', 'output_csv=joinpath(output_dir, "nb_compare_dict.csv")')
            full_source = full_source.replace('output_csv=joinpath(@__DIR__, "..", "outputs", "function_tour") * ""nb_compare.csv""', 'output_csv=joinpath(output_dir, "nb_compare.csv")')
            full_source = full_source.replace('output_csv=joinpath(@__DIR__, "..", "outputs", "function_tour") * ""nb_compare_dict.csv""', 'output_csv=joinpath(output_dir, "nb_compare_dict.csv")')
            # Also replace the pattern with single quotes inside double quotes? We'll also replace the pattern that has * "/file.csv"
            full_source = full_source.replace('output_csv=joinpath(@__DIR__, "..", "outputs", "function_tour") * "/nb_compare.csv"', 'output_csv=joinpath(output_dir, "nb_compare.csv")')
            full_source = full_source.replace('output_csv=joinpath(@__DIR__, "..", "outputs", "function_tour") * "/nb_compare_dict.csv"', 'output_csv=joinpath(output_dir, "nb_compare_dict.csv")')
            # Split back into lines
            cell['source'] = full_source.splitlines(keepends=True)
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)