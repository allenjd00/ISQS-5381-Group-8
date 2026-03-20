import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# Files that are one-row-per-UNITID and safe for direct merging.
DIRECT_MERGE_FILES = {
    "hd": "hd2024.csv",
    "ic": "ic2024.csv",
    "flags": "flags2024.csv",
    "adm": "adm2024.csv",
    "sfa": "sfa2324.csv",
    "cost1": "cost1_2024.csv",
    "cost2": "cost2_2024.csv",
}

DERIVED_MERGE_FILES = {
    "longfmt": OUT / "longformat_unitid_features.csv",
}

# Files currently in long format (many rows per UNITID). These are intentionally
# skipped in this base table to preserve one-row-per-institution shape.
SKIPPED_LONG_FORMAT_FILES = {
    "ef": "ef2024a.csv",
    "gr": "gr2024.csv",
}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required source file: {path}")
    return pd.read_csv(path, low_memory=False)


datasets = {
    name: load_csv(ROOT / file_name) for name, file_name in DIRECT_MERGE_FILES.items()
}
derived_datasets = {
    name: load_csv(path)
    for name, path in DERIVED_MERGE_FILES.items()
    if path.exists()
}

required = ["UNITID"]
for name, df in {**datasets, **derived_datasets}.items():
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

# Prefix non-key columns for provenance clarity

def prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    renamed = {
        col: f"{prefix}_{col}" for col in df.columns if col != "UNITID"
    }
    return df.rename(columns=renamed)

prefixed = {
    name: prefix_columns(df, name)
    for name, df in {**datasets, **derived_datasets}.items()
}

# Build analysis base with HD as the institutional frame
analysis_base = prefixed["hd"].copy()
merge_status_columns = {}

for name in ["ic", "flags", "adm", "sfa", "cost1", "cost2"]:
    merge_col = f"_merge_{name}"
    analysis_base = analysis_base.merge(
        prefixed[name],
        on="UNITID",
        how="left",
        indicator=merge_col,
    )
    merge_status_columns[name] = merge_col

for name in derived_datasets.keys():
    merge_col = f"_merge_{name}"
    analysis_base = analysis_base.merge(
        prefixed[name],
        on="UNITID",
        how="left",
        indicator=merge_col,
    )
    merge_status_columns[name] = merge_col

# Join diagnostics
join_report = {
    "rows": {
        name: int(len(df)) for name, df in datasets.items()
    } | {
        name: int(len(df)) for name, df in derived_datasets.items()
    } | {
        "analysis_base": int(len(analysis_base)),
    },
    "unique_unitid": {
        name: int(df["UNITID"].nunique(dropna=True)) for name, df in datasets.items()
    } | {
        name: int(df["UNITID"].nunique(dropna=True)) for name, df in derived_datasets.items()
    } | {
        "analysis_base": int(analysis_base["UNITID"].nunique(dropna=True)),
    },
    "merge_status_counts": {
        name: analysis_base[col].value_counts(dropna=False).to_dict()
        for name, col in merge_status_columns.items()
    },
    "skipped_long_format_files": {
        name: file_name for name, file_name in SKIPPED_LONG_FORMAT_FILES.items() if (ROOT / file_name).exists()
    },
}

# Save outputs
analysis_csv = OUT / "analysis_base.csv"
analysis_base.to_csv(analysis_csv, index=False)

with open(OUT / "analysis_base_join_report.json", "w", encoding="utf-8") as fp:
    json.dump(join_report, fp, indent=2)

print("Wrote:")
print(f"- {analysis_csv}")
print(f"- {OUT / 'analysis_base_join_report.json'}")
print(f"Shape: {analysis_base.shape}")
