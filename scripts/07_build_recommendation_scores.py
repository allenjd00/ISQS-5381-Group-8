import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

ANALYSIS_BASE = OUT / "analysis_base.csv"
MODELING_READY = OUT / "modeling_ready.csv"
PROJECT_TARGET = OUT / "project_target_unitid.csv"

if not ANALYSIS_BASE.exists():
    raise FileNotFoundError("analysis_base.csv not found. Run scripts/02_build_analysis_base.py first.")
if not MODELING_READY.exists():
    raise FileNotFoundError("modeling_ready.csv not found. Run scripts/03_build_modeling_ready.py first.")
if not PROJECT_TARGET.exists():
    raise FileNotFoundError("project_target_unitid.csv not found. Run scripts/05_build_project_target.py first.")

analysis_base = pd.read_csv(ANALYSIS_BASE, low_memory=False)
modeling_ready = pd.read_csv(MODELING_READY, low_memory=False)
project_target = pd.read_csv(PROJECT_TARGET, low_memory=False)

for frame in (analysis_base, modeling_ready, project_target):
    frame.replace({-3: np.nan, -2: np.nan, -1: np.nan, "-3": np.nan, "-2": np.nan, "-1": np.nan}, inplace=True)

base = analysis_base[[col for col in ["UNITID", "hd_INSTNM", "hd_CITY", "hd_STABBR", "hd_LOCALE", "hd_OBEREG", "ic_STUSRV3", "ic_STUSRV4"] if col in analysis_base.columns]].copy()
base = base.merge(modeling_ready[[col for col in modeling_ready.columns if col == "UNITID" or col.startswith(("adm_", "sfa_", "cost1_", "cost2_", "longfmt_ef_"))]], on="UNITID", how="left")
base = base.merge(project_target[[col for col in ["UNITID", "gr_completion_rate_150", "y_grad_outcome_high"] if col in project_target.columns]], on="UNITID", how="left")


def minmax_scale(series: pd.Series, invert: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    finite = s[np.isfinite(s)]
    if finite.empty:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    min_v = float(finite.min())
    max_v = float(finite.max())
    if max_v == min_v:
        scaled = pd.Series(np.full(len(s), 0.5), index=s.index)
    else:
        scaled = (s - min_v) / (max_v - min_v)
    if invert:
        scaled = 1 - scaled
    return scaled.clip(0, 1)


def mean_or_neutral(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    present = [col for col in cols if col in df.columns]
    if not present:
        return pd.Series(np.full(len(df), 0.5), index=df.index)
    temp = df[present].apply(pd.to_numeric, errors="coerce")
    row_mean = temp.mean(axis=1, skipna=True)
    return row_mean.fillna(0.5)


academic_sources = [
    "gr_completion_rate_150",
    "adm_ADMSSN",
    "adm_ENRLT",
]

cost_sources = [
    "cost1_TUITION1",
    "cost1_FEE1",
    "cost1_ROOMCAP",
    "cost2_NPIS42",
    "cost2_COAIST42",
    "cost2_NPGRN2",
]

career_sources = [
    "ic_STUSRV3",
    "ic_STUSRV4",
    "gr_completion_rate_150",
]

# Academic: combine completion rate + admissions/enrollment strength proxies
academic_parts = []
if "gr_completion_rate_150" in base.columns:
    academic_parts.append(minmax_scale(base["gr_completion_rate_150"]))
if "adm_ADMSSN" in base.columns:
    academic_parts.append(minmax_scale(base["adm_ADMSSN"]))
if "adm_ENRLT" in base.columns:
    academic_parts.append(minmax_scale(base["adm_ENRLT"]))
if academic_parts:
    base["score_academic_quality"] = pd.concat(academic_parts, axis=1).mean(axis=1).fillna(0.5)
else:
    base["score_academic_quality"] = 0.5

# Cost: lower is better
cost_parts = []
for col in [c for c in cost_sources if c in base.columns]:
    cost_parts.append(minmax_scale(base[col], invert=True))
if cost_parts:
    base["score_cost_affordability"] = pd.concat(cost_parts, axis=1).mean(axis=1).fillna(0.5)
else:
    base["score_cost_affordability"] = 0.5

# Career outcomes proxy (services + completion)
career_parts = []
if "ic_STUSRV3" in base.columns:
    career_parts.append(minmax_scale(base["ic_STUSRV3"]))
if "ic_STUSRV4" in base.columns:
    career_parts.append(minmax_scale(base["ic_STUSRV4"]))
if "gr_completion_rate_150" in base.columns:
    career_parts.append(minmax_scale(base["gr_completion_rate_150"]))
if career_parts:
    base["score_career_proxy"] = pd.concat(career_parts, axis=1).mean(axis=1).fillna(0.5)
else:
    base["score_career_proxy"] = 0.5

# Location and safety proxies are neutral in base; app personalizes location.
base["score_location_fit_base"] = 0.5
base["score_safety_qol_proxy"] = 0.5

# Optional interpretable cost estimate for filtering in app
cost_estimate_cols = [c for c in ["cost2_COAIST42", "cost2_NPIS42", "cost1_TUITION1"] if c in base.columns]
if cost_estimate_cols:
    base["estimated_annual_cost"] = base[cost_estimate_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
else:
    base["estimated_annual_cost"] = np.nan

# Default weights from class discussion source (normalized)
raw_weights = {
    "score_academic_quality": 0.74,
    "score_cost_affordability": 0.67,
    "score_career_proxy": 0.73,
    "score_location_fit_base": 0.47,
    "score_safety_qol_proxy": 0.35,
}
weight_sum = sum(raw_weights.values())
weights = {k: v / weight_sum for k, v in raw_weights.items()}

base["score_composite_default"] = (
    base["score_academic_quality"] * weights["score_academic_quality"]
    + base["score_cost_affordability"] * weights["score_cost_affordability"]
    + base["score_career_proxy"] * weights["score_career_proxy"]
    + base["score_location_fit_base"] * weights["score_location_fit_base"]
    + base["score_safety_qol_proxy"] * weights["score_safety_qol_proxy"]
)

ranked = base.sort_values("score_composite_default", ascending=False).copy()
ranked["rank_default"] = np.arange(1, len(ranked) + 1)

if "hd_INSTNM" in ranked.columns and "hd_CITY" in ranked.columns and "hd_STABBR" in ranked.columns:
    ranked["school_display_name"] = (
        ranked["hd_INSTNM"].astype(str)
        + " ("
        + ranked["hd_CITY"].astype(str)
        + ", "
        + ranked["hd_STABBR"].astype(str)
        + ")"
    )

output_cols = [
    "rank_default",
    "UNITID",
    "school_display_name",
    "hd_INSTNM",
    "hd_CITY",
    "hd_STABBR",
    "estimated_annual_cost",
    "score_academic_quality",
    "score_cost_affordability",
    "score_career_proxy",
    "score_location_fit_base",
    "score_safety_qol_proxy",
    "score_composite_default",
]
output_cols = [col for col in output_cols if col in ranked.columns]

scores_path = OUT / "recommendation_base_scores.csv"
ranked[output_cols].to_csv(scores_path, index=False)

report = {
    "source_files": ["analysis_base.csv", "modeling_ready.csv", "project_target_unitid.csv"],
    "row_count": int(len(ranked)),
    "weights_raw": raw_weights,
    "weights_normalized": weights,
    "component_notes": {
        "score_academic_quality": "Completion rate plus admissions/enrollment proxies",
        "score_cost_affordability": "Inverse-scaled tuition/fees/net-price proxies",
        "score_career_proxy": "Proxy only (services + completion), not direct placement outcomes",
        "score_location_fit_base": "Neutral baseline; personalized in app",
        "score_safety_qol_proxy": "Neutral proxy in current IPEDS-only build",
    },
}

report_path = OUT / "recommendation_scoring_report.json"
with report_path.open("w", encoding="utf-8") as fp:
    json.dump(report, fp, indent=2)

print("Wrote:")
print(f"- {scores_path}")
print(f"- {report_path}")
print(f"recommendation rows: {len(ranked)}")
