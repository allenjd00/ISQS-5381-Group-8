import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

GR_FILE = ROOT / "gr2024.csv"

if not GR_FILE.exists():
    raise FileNotFoundError("Missing gr2024.csv required for project target")

gr = pd.read_csv(GR_FILE, low_memory=False)
required = ["UNITID", "CHRTSTAT", "GRTOTLT"]
missing = [column for column in required if column not in gr.columns]
if missing:
    raise ValueError(f"gr2024.csv missing required columns: {missing}")

gr["CHRTSTAT"] = pd.to_numeric(gr["CHRTSTAT"], errors="coerce")
gr["GRTOTLT"] = pd.to_numeric(gr["GRTOTLT"], errors="coerce").replace({-3: np.nan, -2: np.nan, -1: np.nan})

# IPEDS CHRTSTAT labels used:
# 12 = Adjusted cohort (denominator)
# 13 = Completers within 150% of normal time (numerator)
adjusted = (
    gr.loc[gr["CHRTSTAT"] == 12]
    .groupby("UNITID", dropna=False)["GRTOTLT"]
    .sum(min_count=1)
    .rename("gr_adjusted_cohort_total")
)
completers_150 = (
    gr.loc[gr["CHRTSTAT"] == 13]
    .groupby("UNITID", dropna=False)["GRTOTLT"]
    .sum(min_count=1)
    .rename("gr_completers_150_total")
)

target = pd.concat([adjusted, completers_150], axis=1).reset_index()

target["gr_completion_rate_150"] = np.where(
    target["gr_adjusted_cohort_total"] > 0,
    target["gr_completers_150_total"] / target["gr_adjusted_cohort_total"],
    np.nan,
)

valid_rate = target["gr_completion_rate_150"].dropna()
if valid_rate.empty:
    raise ValueError("No valid gr_completion_rate_150 values could be calculated")

median_cut = float(valid_rate.median())
target["y_grad_outcome_high"] = np.where(
    target["gr_completion_rate_150"].notna(),
    (target["gr_completion_rate_150"] >= median_cut).astype(int),
    np.nan,
)

target_path = OUT / "project_target_unitid.csv"
target.to_csv(target_path, index=False)

report = {
    "source": "gr2024.csv",
    "target_definition": {
        "continuous": "gr_completion_rate_150 = sum(CHRTSTAT=13 GRTOTLT) / sum(CHRTSTAT=12 GRTOTLT)",
        "binary": "y_grad_outcome_high = 1 if gr_completion_rate_150 >= median(valid rates), else 0",
        "median_cutoff": median_cut,
    },
    "row_counts": {
        "total_rows": int(len(target)),
        "with_valid_rate": int(target["gr_completion_rate_150"].notna().sum()),
        "with_binary_target": int(target["y_grad_outcome_high"].notna().sum()),
    },
    "target_distribution": {
        str(int(key)): int(value)
        for key, value in target["y_grad_outcome_high"].dropna().value_counts().to_dict().items()
    },
}

with (OUT / "project_target_report.json").open("w", encoding="utf-8") as fp:
    json.dump(report, fp, indent=2)

print("Wrote:")
print(f"- {target_path}")
print(f"- {OUT / 'project_target_report.json'}")
print(f"target rows: {len(target)}")