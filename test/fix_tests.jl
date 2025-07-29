# Script to fix test files
using Base: replace

function fix_test_file(filepath)
    content = read(filepath, String)
    
    # Remove show_plots=false parameter and add bounds where needed for run_single_fit
    content = replace(content, 
        r"run_single_fit\(([^;]+);\s*show_plots=false,\s*show_stats=false\)" => 
        s"run_single_fit(\1; bounds=[(0.01, 2.0), (10.0, 100.0)], show_stats=false)")
    
    content = replace(content, 
        r"run_single_fit\(([^;]+);\s*bounds=([^,]+),\s*show_plots=false,\s*show_stats=false\)" => 
        s"run_single_fit(\1; bounds=\2, show_stats=false)")
    
    content = replace(content,
        r"run_single_fit\(([^;]+);\s*fixed_params=([^,]+),\s*show_plots=false,\s*show_stats=false\)" =>
        s"run_single_fit(\1; fixed_params=\2, bounds=[(0.01, 2.0), (10.0, 100.0)], show_stats=false)")
    
    content = replace(content,
        r"run_single_fit\(([^;]+);\s*model=([^,]+),\s*show_plots=false,\s*show_stats=false\)" =>
        s"run_single_fit(\1; model=\2, bounds=[(0.01, 2.0), (10.0, 100.0)], show_stats=false)")
    
    # Remove show_plots from other functions
    content = replace(content, r";\s*show_plots=false" => "")
    content = replace(content, r"show_plots=false,\s*" => "")
    content = replace(content, r"show_plots=false" => "")
    
    # Fix specific issues
    content = replace(content, "@test_throws BoundsError run_single_fit(x[1:end-1], y, [0.1, 50.0])" =>
                     "@test_throws BoundsError run_single_fit(x[1:end-1], y, [0.1, 50.0])")
    content = replace(content, "@test_throws BoundsError run_single_fit(Float64[], Float64[], [0.1, 50.0])" =>
                     "@test_throws BoundsError run_single_fit(Float64[], Float64[], [0.1, 50.0])")
    
    write(filepath, content)
    println("Fixed: $filepath")
end

# Fix test files
fix_test_file("test/test_fitting.jl")
fix_test_file("test/test_analysis.jl")
