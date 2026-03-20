from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

ANALYSIS_BASE = OUT / "analysis_base.csv"
if not ANALYSIS_BASE.exists():
    raise FileNotFoundError("analysis_base.csv not found. Run scripts/02_build_analysis_base.py first.")

PROJECT_TARGET = OUT / "project_target_unitid.csv"
if not PROJECT_TARGET.exists():
    raise FileNotFoundError("project_target_unitid.csv not found. Run scripts/05_build_project_target.py first.")

# --- Load analysis base ---
base = pd.read_csv(ANALYSIS_BASE, low_memory=False)
project_target = pd.read_csv(PROJECT_TARGET, low_memory=False)

if "UNITID" not in project_target.columns or "y_grad_outcome_high" not in project_target.columns:
    raise ValueError("project_target_unitid.csv must include UNITID and y_grad_outcome_high")

target_cols = [
    "UNITID",
    "gr_adjusted_cohort_total",
    "gr_completers_150_total",
    "gr_completion_rate_150",
    "y_grad_outcome_high",
]
target_cols = [column for column in target_cols if column in project_target.columns]
base = base.merge(project_target[target_cols], on="UNITID", how="left")

# --- Load metadata from data dictionaries ---
def load_metadata(xlsx_path: Path, source_key: str) -> pd.DataFrame:
    varlist = pd.read_excel(xlsx_path, sheet_name="Varlist")
    desc = pd.read_excel(xlsx_path, sheet_name="Description")

    varlist = varlist[["varName", "DataType", "varTitle"]].copy()
    desc = desc[["varName", "longDescription"]].copy()

    md = varlist.merge(desc, on="varName", how="left")
    md["source"] = source_key
    return md

metadata_files = [
    (ROOT / "hd2024.xlsx", "hd"),
    (ROOT / "ic2024.xlsx", "ic"),
    (ROOT / "flags2024.xlsx", "flags"),
    (ROOT / "adm2024.xlsx", "adm"),
    (ROOT / "sfa2324.xlsx", "sfa"),
    (ROOT / "COST1_2024.xlsx", "cost1"),
    (ROOT / "cost2_2024.xlsx", "cost2"),
]

metadata_frames = [
    load_metadata(path, source)
    for path, source in metadata_files
    if path.exists()
]
metadata = pd.concat(metadata_frames, ignore_index=True)

meta_lookup = {
    (r["source"], r["varName"]): {
        "DataType": r["DataType"],
        "varTitle": r["varTitle"],
        "longDescription": r["longDescription"],
    }
    for _, r in metadata.iterrows()
}

# --- Feature shortlist for first-pass model ---
candidate_features = [
    "hd_CONTROL",
    "hd_SECTOR",
    "hd_LOCALE",
    "hd_OBEREG",
    "hd_INSTSIZE",
    "hd_C21BASIC",
    "hd_CARNEGIEIC",
    "hd_CARNEGIERSCH",
    "hd_HBCU",
    "hd_TRIBAL",
    "hd_MEDICAL",
    "hd_HOSPITAL",
    "hd_LANDGRNT",
    "hd_CBSA",
    "hd_CBSATYPE",
    "hd_CSA",
    "hd_LONGITUD",
    "hd_LATITUDE",
    "adm_APPLCN",
    "adm_ADMSSN",
    "adm_ENRLT",
    "sfa_UAGRNTP",
    "sfa_UPGRNTP",
    "sfa_UFLOANP",
    "cost1_TUITION1",
    "cost1_FEE1",
    "cost1_TUITION2",
    "cost1_FEE2",
    "cost1_TUITION3",
    "cost1_FEE3",
    "cost1_ROOMCAP",
    "cost2_NPIS42",
    "cost2_COAIST42",
    "cost2_NPGRN2",
    "longfmt_ef_rows",
    "longfmt_ef_eftotlt_max",
    "longfmt_ef_eftotlt_mean",
]

present_features = [c for c in candidate_features if c in base.columns]

required_cols = ["UNITID", "y_grad_outcome_high"] + present_features
missing_required = [c for c in required_cols if c not in base.columns]
if missing_required:
    raise ValueError(f"Required columns missing in analysis_base.csv: {missing_required}")

work = base[required_cols].copy()

# Replace common IPEDS sentinel values
sentinel_values = {
    -3: np.nan,
    -2: np.nan,
    -1: np.nan,
    "-3": np.nan,
    "-2": np.nan,
    "-1": np.nan,
}
work = work.replace(sentinel_values)

# Binary target from GR outcome build step
work["y_grad_outcome_high"] = pd.to_numeric(work["y_grad_outcome_high"], errors="coerce")
modeling_ready = work.dropna(subset=["y_grad_outcome_high"]).copy()
modeling_ready["y_grad_outcome_high"] = modeling_ready["y_grad_outcome_high"].astype(int)

# --- Build variable map for all analysis_base columns ---
selected_feature_set = set(present_features)
rows = []
for col in base.columns:
    if col == "UNITID":
        role = "id"
        selected = 1
        source = "key"
        source_var = "UNITID"
        dtype = "identifier"
        title = "Institution identifier"
        long_desc = "Unique institution identifier used for joins across files."
        reason = "Primary join key and row identifier."
    elif col == "y_grad_outcome_high":
        role = "target_raw"
        selected = 1
        source = "gr"
        source_var = "y_grad_outcome_high"
        dtype = "binary_int"
        title = "Graduation outcome high indicator (derived)"
        long_desc = "Derived from gr_completion_rate_150 median split in scripts/05_build_project_target.py."
        reason = "Primary project-aligned supervised target."
    elif col in selected_feature_set:
        role = "feature"
        selected = 1
        prefix, source_var = col.split("_", 1)
        source = prefix
        md = meta_lookup.get((source, source_var), {})
        dtype = md.get("DataType", "")
        title = md.get("varTitle", "")
        long_desc = md.get("longDescription", "")
        reason = "Selected for first-pass baseline model." 
    else:
        role = "excluded"
        selected = 0
        if col.startswith("_merge_"):
            source = "merge"
            source_var = col
            dtype = "join_indicator"
            title = "Join status indicator"
            long_desc = "Merge indicator generated during analysis base construction."
        elif "_" in col:
            prefix, source_var = col.split("_", 1)
            source = prefix
            md = meta_lookup.get((source, source_var), {})
            dtype = md.get("DataType", "")
            title = md.get("varTitle", "")
            long_desc = md.get("longDescription", "")
        else:
            source = "derived"
            source_var = col
            dtype = ""
            title = ""
            long_desc = ""
        reason = "Not in first-pass baseline scope."

    rows.append(
        {
            "analysis_column": col,
            "source": source,
            "source_var": source_var,
            "role": role,
            "selected_for_modeling_ready": selected,
            "data_type": dtype,
            "var_title": title,
            "long_description": long_desc,
            "selection_reason": reason,
        }
    )

# Add derived target row to mapping
rows.append(
    {
        "analysis_column": "gr_completion_rate_150",
        "source": "gr",
        "source_var": "CHRTSTAT_13_over_12",
        "role": "target",
        "selected_for_modeling_ready": 1,
        "data_type": "float",
        "var_title": "Completion rate within 150% (derived)",
        "long_description": "sum(CHRTSTAT=13 GRTOTLT) / sum(CHRTSTAT=12 GRTOTLT) from gr2024.",
        "selection_reason": "Continuous project-aligned outcome metric.",
    }
)

rows.append(
    {
        "analysis_column": "y_grad_outcome_high",
        "source": "gr",
        "source_var": "CHRTSTAT_13_over_12_median_split",
        "role": "target",
        "selected_for_modeling_ready": 1,
        "data_type": "binary_int",
        "var_title": "High graduation outcome indicator (derived)",
        "long_description": "1 if gr_completion_rate_150 >= median(valid rates), else 0.",
        "selection_reason": "Primary supervised-learning target for baseline classification.",
    }
)

variable_map = pd.DataFrame(rows)

# Save outputs
variable_map_path = OUT / "variable_map.csv"
modeling_ready_path = OUT / "modeling_ready.csv"

variable_map.to_csv(variable_map_path, index=False)
modeling_ready.to_csv(modeling_ready_path, index=False)

print("Wrote:")
print(f"- {variable_map_path}")
print(f"- {modeling_ready_path}")
print(f"modeling_ready shape: {modeling_ready.shape}")
print("target distribution (y_grad_outcome_high):")
print(modeling_ready["y_grad_outcome_high"].value_counts(dropna=False).to_string())
