# GrowthParameterEstimation.jl showcase

The showcase fits ten published, public tumour-growth datasets with one set of package calls and one model-selection rule. The datasets span cell culture, spheroids, mice and patients. The results are published as a separate documentation site (`showcase/docs/`).

It is kept apart from the package: it has its own environments, and its data and fits never run in the package test suite.

## Datasets

See [`data/INVENTORY.md`](data/INVENTORY.md) for counts checked against the downloaded files, and [`data/manifest.toml`](data/manifest.toml) for sources, licences and SHA-256 hashes.

| Scale | Datasets |
| --- | --- |
| Cells | Johnson 2019 BT-474 (low-density growth), Kaznatcheev 2019 H3122 (parental vs resistant co-culture, alectinib), Harris 2016 PC9 (erlotinib) |
| Spheroids | Browning 2021 melanoma spheroids, Murphy 2023 spheroids under oxygen switches |
| Mice | Benzekry 2014 Lewis lung carcinoma, Benzekry 2014 LM2-4LUC+ breast, Ebos lab sunitinib/axitinib breast xenografts |
| Patients | Ghaffari Laleh 2022 RECIST diameters, Bruchovsky intermittent androgen deprivation PSA |

## Data policy

No third-party data is committed. `data/prepare.py` downloads each file from its original source, checks its hash, and writes tidy CSVs to `data/tidy/` (git-ignored). Only derived results (fitted parameters, model rankings, figures) are committed under `results/` and `figures/`. Each dataset page cites the original paper and states the source licence.

## Rebuild

```bash
python3 -m pip install -r showcase/data/requirements.txt
python3 showcase/data/prepare.py          # download, verify, tidy, write INVENTORY.md
```

Fitting scripts and the documentation build are added in later milestones (see the project plan).
