"""Download the showcase datasets, check their hashes and convert them to one tidy schema.

Usage (from the repository root):

    python3 showcase/data/prepare.py            # download, verify, convert, write INVENTORY.md
    python3 showcase/data/prepare.py --offline  # reuse files already in showcase/data/raw/

Raw files go to showcase/data/raw/ and tidy CSVs to showcase/data/tidy/. Both folders are
git-ignored: third-party data is never committed. Only INVENTORY.md (counts, no data) is.

Tidy schema, one row per observation:
    dataset, subject, condition, time, time_unit, value, value_unit, dose, dose_unit,
    treatment_on, cell_line, seed, replicate

Requires: pandas, numpy, xlrd (for .xls), openpyxl (for .xlsx).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import re
import sys
import tomllib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
TIDY = HERE / "tidy"
COLUMNS = [
    "dataset", "subject", "condition", "time", "time_unit", "value", "value_unit",
    "dose", "dose_unit", "treatment_on", "cell_line", "seed", "replicate",
]


# --------------------------------------------------------------------------- download

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(dataset_id: str, entry: dict, offline: bool) -> Path:
    dest = RAW / dataset_id / entry["name"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        if offline:
            raise FileNotFoundError(f"{dest} missing and --offline given")
        req = urllib.request.Request(entry["url"], headers={"User-Agent": "GrowthParameterEstimation-showcase"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            dest.write_bytes(resp.read())
    got = sha256(dest)
    if got != entry["sha256"]:
        raise ValueError(f"hash mismatch for {dest}: expected {entry['sha256']}, got {got}")
    return dest


def frame(**cols) -> pd.DataFrame:
    df = pd.DataFrame(cols)
    for c in COLUMNS:
        if c not in df:
            df[c] = pd.NA
    return df[COLUMNS]


# --------------------------------------------------------------------------- adapters

def johnson(files: dict[str, Path]) -> pd.DataFrame:
    """BT-474 counts per well. Seeding numbers follow code/load_BT474.m in the source repo.

    Each sheet: first row = well names, first column = time (h), last row = quality flag
    (0 die-off, 2 persist, 3 bad fit, other = normal). Rows with flag 3 are dropped, as in the paper.
    """
    seeds = {
        "BT_474_Nseed5_5_30wflags.xls": 5, "BT_474_Nseed2_7_1wflags.xls": 2,
        "BT_474_Nseed10_7_12wflags.xls": 10, "BT_474_Nseed67_30wflags.xls": 6,
        "BT_474_Nseed15802wflags.xls": 15, "BT-474_Nseed12_8_16wflags.xls": 12,
        "BT-474_Nseed8_8_16wflags.xls": 8,
    }
    out = []
    for name, path in files.items():
        raw = pd.read_excel(path, header=None)
        wells = [re.sub(r".*:\s*", "", str(w)).strip() for w in raw.iloc[0, 1:]]
        body = raw.iloc[1:].reset_index(drop=True)
        has_flags = name != "BT-474_Nseed_large.xls"
        flags = body.iloc[-1, 1:].tolist() if has_flags else [None] * len(wells)
        data = body.iloc[:-1] if has_flags else body
        t = pd.to_numeric(data.iloc[:, 0], errors="coerce").round()
        for j, well in enumerate(wells):
            if has_flags and flags[j] == 3:
                continue
            if name == "BT-474_Nseed_large.xls":
                seed = 512 if well[0] in "BCD" else 1024
            else:
                seed = seeds[name]
            y = pd.to_numeric(data.iloc[:, j + 1], errors="coerce")
            ok = t.notna() & y.notna()
            out.append(frame(
                dataset="johnson2019_bt474", subject=f"{name.split('.')[0]}:{well}",
                condition=f"seed{seed}", time=t[ok].values, time_unit="h", value=y[ok].values,
                value_unit="cells", dose=0.0, treatment_on=False, cell_line="BT-474", seed=seed,
                replicate=well,
            ))
    return pd.concat(out, ignore_index=True)


def _literal_array(src: str, name: str):
    m = re.search(rf"^{name}\s*=\s*np\.array\((\[.*?\])\)\s*$", src, re.S | re.M)
    if not m:
        raise ValueError(f"{name} not found in DataLoad.py")
    return np.array(ast.literal_eval(m.group(1).replace("np.NaN", "None").replace("np.nan", "None")), dtype=float)


def kaznatcheev(files: dict[str, Path]) -> pd.DataFrame:
    """Fluorescent area per well over time: red = parental (mCherry), green = resistant (GFP).

    The well layout arrays are read as literals from DataLoad.py; that file is never executed.
    """
    red = np.load(files["red_all.npy"], allow_pickle=False)
    green = np.load(files["green_all.npy"], allow_pickle=False)
    times = np.load(files["time_all.npy"], allow_pickle=False)
    src = files["DataLoad.py"].read_text()
    propP, drug, fibro = (_literal_array(src, n) for n in ("propP", "drugV", "fibroV"))
    stamps = pd.to_datetime({"year": times[:, 0], "month": times[:, 1], "day": times[:, 2], "hour": times[:, 3], "minute": times[:, 4]})
    hours = ((stamps - stamps.iloc[0]).dt.total_seconds() / 3600).values
    rows = "BCDEFGHIJKLMNO"
    out = []
    for i in range(red.shape[1]):
        for j in range(red.shape[2]):
            if np.isnan(propP[i, j]):
                continue
            well = f"{rows[i]}{j + 2}"
            cond = f"parental{propP[i, j]:.3g}_{'alectinib' if drug[i, j] else 'DMSO'}{'_CAF' if fibro[i, j] else ''}"
            for series, arr, line in (("parental", red, "H3122 parental"), ("resistant", green, "H3122 alectinib-resistant")):
                out.append(frame(
                    dataset="kaznatcheev2019_h3122", subject=f"{well}:{series}", condition=cond,
                    time=hours, time_unit="h", value=arr[:, i, j], value_unit="fluorescent area (px)",
                    treatment_on=bool(drug[i, j]),
                    cell_line=line, seed=propP[i, j], replicate=well,
                ))
    return pd.concat(out, ignore_index=True)


def harris(files: dict[str, Path]) -> pd.DataFrame:
    a = pd.read_csv(files["DS345 growth curve data.csv"])
    a = frame(
        dataset="harris2016_pc9", subject="growth:" + a["UID"].astype(str), condition=a["Subline"] + "_" + a["Condition"],
        time=a["Time_h"], time_unit="h", value=a["Cell.count"], value_unit="cells",
        treatment_on=a["Condition"].eq("Erlotinib"), cell_line=a["Subline"], replicate=a["Well"],
    )
    b = pd.read_csv(files["DS345+erl in HTS core.csv"])
    b = frame(
        dataset="harris2016_pc9", subject="dose:" + b["cellLine"] + "_" + b["ID"].astype(str), condition=b["cellLine"] + "_erl" + b["conc"].astype(str),
        time=b["time"], time_unit="h", value=b["NumNuclei"], value_unit="nuclei",
        dose=b["conc"], dose_unit="uM erlotinib", treatment_on=b["conc"] > 0, cell_line=b["cellLine"], replicate=b["ID"],
    )
    return pd.concat([a, b], ignore_index=True)


def browning(files: dict[str, Path]) -> pd.DataFrame:
    d = pd.read_csv(files["IncucyteData.csv"])
    d["rep"] = d.groupby(["CellLine", "InitialCondition", "Day"]).cumcount() + 1
    return frame(
        dataset="browning2021_spheroids", subject=d["CellLine"] + "_" + d["InitialCondition"].astype(str) + "_s" + d["rep"].astype(str),
        condition=d["CellLine"] + "_seed" + d["InitialCondition"].astype(str), time=d["Day"], time_unit="d",
        value=d["R"], value_unit="outer radius (um)", dose=0.0, treatment_on=False, cell_line=d["CellLine"],
        seed=d["InitialCondition"], replicate=d["rep"],
    )


def murphy(files: dict[str, Path]) -> pd.DataFrame:
    """Fixed spheroids: one radius per spheroid per day (cross-sectional, not longitudinal),
    so each condition is one curve with several spheroids per day.

    Condition codes follow the source repository's 2Code scripts; their oxygen schedules are
    decoded in the analysis step.
    """
    out = []
    for path in files.values():
        d = pd.read_excel(path)
        d = d[d.get("Keep", 1) == 1]
        out.append(frame(
            dataset="murphy2023_oxygen", subject=d["CellLine"] + "_cond" + d["Condition"].astype(str), condition=d["CellLine"] + "_cond" + d["Condition"].astype(str),
            time=d["Day"], time_unit="d", value=d["OuterRadius"], value_unit="outer radius (um)", cell_line=d["CellLine"],
            replicate=d["Replicate"],
        ))
    return pd.concat(out, ignore_index=True)


def zenodo_long(dataset: str, unit: str, line: str):
    def adapter(files: dict[str, Path]) -> pd.DataFrame:
        (path,) = files.values()
        d = pd.read_csv(path, sep=None, engine="python")
        d.columns = ["ID", "Time", "Value"]
        return frame(
            dataset=dataset, subject=d["ID"].astype(str), condition="control", time=d["Time"], time_unit="d",
            value=d["Value"], value_unit=unit, dose=0.0, treatment_on=False, cell_line=line, replicate=d["ID"],
        )
    return adapter


def ebos(files: dict[str, Path]) -> pd.DataFrame:
    """Wide sheet: each experiment block starts with a 'Days Post-Implatation' header row.

    Columns 11..35 hold pre-surgical volumes (mm3) on the days in that header row. Treatment
    windows are given per group; their start days are taken from table_summary.pdf in the analysis step.
    """
    raw = pd.read_excel(files["ebos_data_breast.xlsx"], header=None)
    headers = raw.index[raw.iloc[:, 1].astype(str).str.startswith("Days Post")].tolist() + [len(raw)]
    out = []
    for h, nxt in zip(headers[:-1], headers[1:]):
        exp = str(raw.iloc[h, 0])
        days = pd.to_numeric(raw.iloc[h, 11:36], errors="coerce")
        cols = [c for c, v in days.items() if pd.notna(v)]
        for r in range(h + 1, nxt):
            group, mouse = raw.iloc[r, 1], raw.iloc[r, 10]
            if pd.isna(group) or pd.isna(mouse):
                continue
            vol = pd.to_numeric(raw.iloc[r, cols], errors="coerce")
            ok = vol.notna().values
            out.append(frame(
                dataset="ebos_sunitinib", subject=f"{exp}:{group}:{mouse}", condition=str(group),
                time=days[cols].values[ok], time_unit="d post-implantation", value=vol.values[ok], value_unit="mm3",
                treatment_on=pd.NA, cell_line=exp, replicate=str(mouse),
            ))
    return pd.concat(out, ignore_index=True)


def ghaffari(files: dict[str, Path]) -> pd.DataFrame:
    x = pd.ExcelFile(files["S1_Data.xlsx"])
    out = []
    for sheet in x.sheet_names:
        d = x.parse(sheet)
        d["TargetLesionLongDiam_mm"] = pd.to_numeric(d["TargetLesionLongDiam_mm"], errors="coerce")
        out.append(frame(
            dataset="ghaffarilaleh2022_recist", subject=sheet + ":" + d["Patient_Anonmyized"].astype(str), condition=d["Study_Arm"],
            time=d["Treatment_Day"], time_unit="d", value=d["TargetLesionLongDiam_mm"], value_unit="target lesion longest diameter (mm)",
            treatment_on=d["Treatment_Day"] >= 0, replicate=sheet,
        ))
    return pd.concat(out, ignore_index=True)


def bruchovsky(files: dict[str, Path]) -> pd.DataFrame:
    """Columns per readme.txt: patient, date, CPA, LEU, PSA, testosterone, cycle, on(1)/off(0), day, day."""
    out = []
    with zipfile.ZipFile(files["dataTanaka.zip"]) as z:
        for name in sorted(z.namelist()):
            if "Bruchovsky_et_al/patient" not in name or not name.endswith(".txt"):
                continue
            d = pd.read_csv(io.BytesIO(z.read(name)), header=None)
            d = d.iloc[:, :10]
            d.columns = ["patient", "date", "cpa", "leu", "psa", "testosterone", "cycle", "on", "day_abs", "day"]
            d["psa"] = pd.to_numeric(d["psa"], errors="coerce")
            d = d[d["psa"].notna()]
            out.append(frame(
                dataset="bruchovsky_iadt_psa", subject=d["patient"].astype(str), condition="IADT", time=d["day"] - d["day"].min(),
                time_unit="d", value=d["psa"], value_unit="PSA (ng/mL)", treatment_on=d["on"].astype(float) == 1,
                replicate=d["cycle"],
            ))
    return pd.concat(out, ignore_index=True)


ADAPTERS = {
    "johnson2019_bt474": johnson,
    "kaznatcheev2019_h3122": kaznatcheev,
    "harris2016_pc9": harris,
    "browning2021_spheroids": browning,
    "murphy2023_oxygen": murphy,
    "benzekry2014_llc": zenodo_long("benzekry2014_llc", "mm3", "LLC"),
    "benzekry2014_lm24": zenodo_long("benzekry2014_lm24", "mm3", "LM2-4LUC+"),
    "ebos_sunitinib": ebos,
    "ghaffarilaleh2022_recist": ghaffari,
    "bruchovsky_iadt_psa": bruchovsky,
}


# --------------------------------------------------------------------------- inventory

def summarise(df: pd.DataFrame) -> dict:
    per = df.groupby("subject").size()
    return dict(
        curves=int(per.size), observations=int(len(df)), conditions=int(df["condition"].nunique()),
        pts_min=int(per.min()), pts_median=float(per.median()), pts_max=int(per.max()),
        t_min=float(df["time"].min()), t_max=float(df["time"].max()), time_unit=str(df["time_unit"].iloc[0]),
        value_unit=str(df["value_unit"].iloc[0]),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    manifest = tomllib.loads((HERE / "manifest.toml").read_text())["datasets"]
    TIDY.mkdir(exist_ok=True)
    rows = []
    for ds_id, meta in manifest.items():
        if args.only and ds_id not in args.only:
            continue
        files = {f["name"].split("/")[-1]: fetch(ds_id, f, args.offline) for f in meta["files"]}
        tidy = ADAPTERS[ds_id](files)
        tidy.to_csv(TIDY / f"{ds_id}.csv", index=False)
        s = summarise(tidy)
        rows.append((ds_id, meta, s))
        print(f"{ds_id}: {s['curves']} curves, {s['observations']} observations")
    lines = [
        "# Showcase data inventory",
        "",
        "Generated by `prepare.py` from the files listed in `manifest.toml` (hashes checked).",
        "Counts are after the documented filtering in each adapter. No data is stored in this repository.",
        "",
        "| Dataset | Scale | Treated | Curves | Observations | Points per curve (min / median / max) | Time span | Measured | Licence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ds_id, meta, s in rows:
        lines.append(
            f"| [{meta['title']}]({meta['source']}) | {meta['scale']} | {'yes' if meta['treated'] else 'no'} | {s['curves']} | "
            f"{s['observations']} | {s['pts_min']} / {s['pts_median']:g} / {s['pts_max']} | {s['t_min']:g} to {s['t_max']:g} {s['time_unit']} | "
            f"{s['value_unit']} | {meta['licence']} |"
        )
    (HERE / "INVENTORY.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
