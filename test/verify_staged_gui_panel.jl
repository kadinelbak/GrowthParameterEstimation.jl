include("../examples/gui/pipeline_gui_app.jl")

panel, fig = _staged_panel("examples/gui/data/staged_monoculture.csv")
println("FIG_TYPE=", typeof(fig))
println("HAS_DATA_KEY=", haskey(fig, "data"))
println("HAS_LAYOUT_KEY=", haskey(fig, "layout"))
println("N_TRACES=", haskey(fig, "data") ? length(fig["data"]) : -1)

if haskey(fig, "layout") && haskey(fig["layout"], "title")
    println("TITLE=", fig["layout"]["title"])
end

println("PANEL_TYPE=", typeof(panel))
