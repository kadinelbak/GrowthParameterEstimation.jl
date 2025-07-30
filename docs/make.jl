using GrowthParamEst
using Documenter

DocMeta.setdocmeta!(GrowthParamEst, :DocTestSetup, :(using GrowthParamEst); recursive=true)

makedocs(;
    modules=[GrowthParamEst],
    authors="Kadin",
    sitename="GrowthParamEst.jl",
    format=Documenter.HTML(;
        canonical="https://kadinelbak.github.io/GrowthParamEst.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kadinelbak/V1SimpleODE.jl",
    devbranch="master",
)
