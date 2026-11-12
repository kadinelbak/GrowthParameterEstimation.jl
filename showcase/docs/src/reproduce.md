# Reproduce

```bash
git clone https://github.com/kadinelbak/GrowthParameterEstimation.jl
cd GrowthParameterEstimation.jl
python3 -m pip install -r showcase/data/requirements.txt
python3 -I showcase/data/prepare.py            # download, verify hashes, write tidy CSVs
julia --project=showcase -e 'using Pkg; Pkg.instantiate()'
julia --project=showcase -t auto showcase/run_pipeline.jl --only benzekry2014_llc   # one dataset
julia --project=showcase -t auto showcase/run_pipeline.jl                            # all datasets
julia --project=showcase showcase/docs/make.jl                  # rebuild this site
```

The pipeline writes `showcase/results/<id>/` (CSV and TOML) and `showcase/figures/<id>/` (PNG, SVG, PDF). The site is built only from those committed results, so building it needs no data download and no fitting.

## Your own data

`run_pipeline.jl` reads `showcase/pipeline.toml`. To run the same recipe on your own measurements, copy that file, add a `[[dataset]]` block with `csv = "path/to/file.csv"` and the names of your curve-id, time and value columns, and pass the copy as the first argument:

```bash
julia --project=showcase -t auto showcase/run_pipeline.jl my_pipeline.toml
```

Every key in `[defaults]` (model panel, number of starts, profiles, forecast) can be overridden per dataset.
