using GrowthParameterEstimation
using Documenter

DocMeta.setdocmeta!(GrowthParameterEstimation, :DocTestSetup, :(using GrowthParameterEstimation); recursive=true)

makedocs(;
    modules=[GrowthParameterEstimation],
    authors="Kadin",
    sitename="GrowthParameterEstimation.jl",
    format=Documenter.HTML(;
        canonical="https://kadinelbak.github.io/GrowthParameterEstimation.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kadinelbak/GrowthParameterEstimation.jl",
    devbranch="master",
)
