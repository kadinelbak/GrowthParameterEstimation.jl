using V1SimpleODE
using Documenter

DocMeta.setdocmeta!(V1SimpleODE, :DocTestSetup, :(using V1SimpleODE); recursive=true)

makedocs(;
    modules=[V1SimpleODE],
    authors="Kadin",
    sitename="V1SimpleODE.jl",
    format=Documenter.HTML(;
        canonical="https://kadinelbak.github.io/V1SimpleODE.jl",
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
