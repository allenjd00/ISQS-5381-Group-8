import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

EF_FILE = ROOT / "ef2024a.csv"
GR_FILE = ROOT / "gr2024.csv"


def _sanitize_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    clean = df.copy()
    for col in columns:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.replace({-3: np.nan, -2: np.nan, -1: np.nan})
    return clean


def flatten_ef(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "EFTOTLT",
        "EFTOTLM",
        "EFTOTLW",
        "EFAIANT",
        "EFASIAT",
        "EFBKAAT",
        "EFHISPT",
        "EFNHPIT",
        "EFWHITT",
        "EF2MORT",
    ]
    clean = _sanitize_numeric(df, numeric_cols)
    grouped = clean.groupby("UNITID", dropna=False)

    out = pd.DataFrame({"UNITID": grouped.size().index})
    out["ef_rows"] = grouped.size().values
    for cat_col in ["LINE", "SECTION", "LSTUDY", "EFALEVEL"]:
        if cat_col in clean.columns:
            out[f"ef_{cat_col.lower()}_nunique"] = grouped[cat_col].nunique(dropna=True).values

    for col in numeric_cols:
        if col in clean.columns:
            out[f"ef_{col.lower()}_max"] = grouped[col].max().values
            out[f"ef_{col.lower()}_mean"] = grouped[col].mean().values

    return out


def flatten_gr(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "GRTOTLT",
        "GRTOTLM",
        "GRTOTLW",
        "GRAIANT",
        "GRASIAT",
        "GRBKAAT",
        "GRHISPT",
        "GRNHPIT",
        "GRWHITT",
        "GR2MORT",
    ]
    clean = _sanitize_numeric(df, numeric_cols)
    grouped = clean.groupby("UNITID", dropna=False)

    out = pd.DataFrame({"UNITID": grouped.size().index})
    out["gr_rows"] = grouped.size().values
    for cat_col in ["LINE", "SECTION", "COHORT", "CHRTSTAT", "GRTYPE"]:
        if cat_col in clean.columns:
            out[f"gr_{cat_col.lower()}_nunique"] = grouped[cat_col].nunique(dropna=True).values

    for col in numeric_cols:
        if col in clean.columns:
            out[f"gr_{col.lower()}_max"] = grouped[col].max().values
            out[f"gr_{col.lower()}_mean"] = grouped[col].mean().values

    return out


if not EF_FILE.exists() or not GR_FILE.exists():
    missing = [str(path.name) for path in [EF_FILE, GR_FILE] if not path.exists()]
    raise FileNotFoundError(f"Missing required long-format files: {missing}")

ef = pd.read_csv(EF_FILE, low_memory=False)
gr = pd.read_csv(GR_FILE, low_memory=False)

if "UNITID" not in ef.columns or "UNITID" not in gr.columns:
    raise ValueError("Both ef2024a.csv and gr2024.csv must contain UNITID")

ef_flat = flatten_ef(ef)
gr_flat = flatten_gr(gr)

longfmt = ef_flat.merge(gr_flat, on="UNITID", how="outer", indicator="_merge_ef_gr")

longfmt_path = OUT / "longformat_unitid_features.csv"
longfmt.to_csv(longfmt_path, index=False)

report = {
    "source_rows": {
        "ef2024a": int(len(ef)),
        "gr2024": int(len(gr)),
    },
    "source_unique_unitid": {
        "ef2024a": int(ef["UNITID"].nunique(dropna=True)),
        "gr2024": int(gr["UNITID"].nunique(dropna=True)),
    },
    "flattened_rows": int(len(longfmt)),
    "flattened_unique_unitid": int(longfmt["UNITID"].nunique(dropna=True)),
    "merge_status_counts": longfmt["_merge_ef_gr"].value_counts(dropna=False).to_dict(),
}

with (OUT / "longformat_unitid_features_report.json").open("w", encoding="utf-8") as fp:
    json.dump(report, fp, indent=2)

print("Wrote:")
print(f"- {longfmt_path}")
print(f"- {OUT / 'longformat_unitid_features_report.json'}")
print(f"Shape: {longfmt.shape}")