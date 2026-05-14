#!/usr/bin/env julia
# Test if the GUI app file has any syntax errors

using Pkg
Pkg.activate("examples/gui")

try
    @info "Loading GUI app..."
    include("examples/gui/pipeline_gui_app.jl")
    @info "GUI app loaded successfully - no syntax errors!"
catch err
    @error "Failed to load GUI app: $(err)"
    showerror(stderr, err)
    exit(1)
end
