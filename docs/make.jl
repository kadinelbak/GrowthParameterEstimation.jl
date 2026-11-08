using GrowthParameterEstimation
using Documenter

DocMeta.setdocmeta!(GrowthParameterEstimation, :DocTestSetup, :(using GrowthParameterEstimation); recursive=true)

makedocs(;
    modules=[GrowthParameterEstimation],
    warnonly = [:missing_docs, :cross_references],
    authors="Kadin El bakkouri",
    sitename="GrowthParameterEstimation.jl",
    checkdocs = :none, # allow building while we flesh out full API docs
    format=Documenter.HTML(;
        canonical="https://kadinelbak.github.io/GrowthParameterEstimation.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Tutorials" => [
            "Getting Started" => "tutorial/getting_started.md",
            "Basic Examples" => "tutorial/basic_examples.md",
            "Advanced Usage" => "tutorial/advanced_usage.md",
        ],
        "Identifiability" => "identifiability.md",
        "Control handoff" => "control.md",
        "Research use" => "research_use.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/kadinelbak/GrowthParameterEstimation.jl",
    devbranch="main",
)
