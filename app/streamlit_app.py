from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SCORES_FILE = OUT / "recommendation_base_scores.csv"

st.set_page_config(page_title="College Match MVP", layout="wide")
st.title("College Match MVP")
st.caption("Top-N recommendations from IPEDS-based component scoring")

if not SCORES_FILE.exists():
    st.error("Missing outputs/recommendation_base_scores.csv. Run scripts/07_build_recommendation_scores.py first.")
    st.stop()

base = pd.read_csv(SCORES_FILE, low_memory=False)
required_cols = {
    "UNITID",
    "score_academic_quality",
    "score_cost_affordability",
    "score_career_proxy",
    "score_location_fit_base",
    "score_safety_qol_proxy",
    "score_composite_default",
}
missing = [c for c in required_cols if c not in base.columns]
if missing:
    st.error(f"Scoring file missing required columns: {missing}")
    st.stop()

for col in [
    "score_academic_quality",
    "score_cost_affordability",
    "score_career_proxy",
    "score_location_fit_base",
    "score_safety_qol_proxy",
    "score_composite_default",
    "estimated_annual_cost",
]:
    if col in base.columns:
        base[col] = pd.to_numeric(base[col], errors="coerce")

states = sorted([s for s in base.get("hd_STABBR", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if s.strip()])

career_focus_presets = {
    "Any": [],
    "Business/Management": ["business", "management", "commerce"],
    "Engineering/Technology": ["engineering", "technology", "polytechnic", "tech"],
    "Health/Nursing": ["health", "nursing", "medical", "biomedical"],
    "Education/Teaching": ["education", "teaching", "teachers"],
    "Arts/Design": ["art", "design", "music", "performing"],
    "Culinary/Hospitality": ["culinary", "hospitality", "chef"],
    "Religious/Theology": ["theology", "seminary", "bible", "ministry", "christian"],
}

with st.sidebar:
    st.header("Preferences")
    preferred_state = st.selectbox("Preferred State", options=["Any"] + states, index=0)
    enforce_state_filter = st.checkbox("Require preferred state", value=True)
    max_budget = st.number_input("Max Annual Budget ($)", min_value=0, max_value=200000, value=40000, step=1000)
    require_known_cost = st.checkbox("Require known annual cost", value=True)
    top_n = st.slider("How many results?", min_value=3, max_value=10, value=3)

    st.subheader("Career/Program Fit")
    career_focus = st.selectbox("Career focus preset", options=list(career_focus_presets.keys()), index=0)
    strict_career_filter = st.checkbox("Apply career keywords as strict filter", value=False)
    include_keywords_raw = st.text_input(
        "Include school name keywords (comma-separated)",
        value="",
        placeholder=", ".join(career_focus_presets[career_focus]),
    )
    exclude_keywords_raw = st.text_input(
        "Exclude school name keywords (comma-separated)",
        value="theology, seminary, bible, culinary",
    )

    st.subheader("Factor Weights")
    w_academic = st.slider("Academic Quality", 0.0, 1.0, 0.74, 0.01)
    w_cost = st.slider("Cost & Aid", 0.0, 1.0, 0.67, 0.01)
    w_career = st.slider("Career Outcomes (Proxy)", 0.0, 1.0, 0.73, 0.01)
    w_location = st.slider("Location Fit", 0.0, 1.0, 0.47, 0.01)
    w_safety = st.slider("Safety & Quality of Life (Proxy)", 0.0, 1.0, 0.35, 0.01)

weights_raw = {
    "score_academic_quality": w_academic,
    "score_cost_affordability": w_cost,
    "score_career_proxy": w_career,
    "score_location_fit_base": w_location,
    "score_safety_qol_proxy": w_safety,
}
weight_sum = sum(weights_raw.values())
if weight_sum == 0:
    st.warning("All weights are zero. Resetting to equal weights.")
    weights = {k: 1 / len(weights_raw) for k in weights_raw}
else:
    weights = {k: v / weight_sum for k, v in weights_raw.items()}

data = base.copy()

data["_inst_name"] = data.get("hd_INSTNM", pd.Series("", index=data.index)).astype(str).str.lower()

include_keywords = [keyword.strip().lower() for keyword in include_keywords_raw.split(",") if keyword.strip()]
exclude_keywords = [keyword.strip().lower() for keyword in exclude_keywords_raw.split(",") if keyword.strip()]

preset_keywords = career_focus_presets.get(career_focus, [])
effective_include_keywords = include_keywords if include_keywords else preset_keywords

if effective_include_keywords and strict_career_filter:
    include_mask = pd.Series(False, index=data.index)
    for keyword in effective_include_keywords:
        include_mask = include_mask | data["_inst_name"].str.contains(keyword, na=False)
    data = data.loc[include_mask].copy()

if effective_include_keywords and not strict_career_filter:
    include_mask = pd.Series(False, index=data.index)
    for keyword in effective_include_keywords:
        include_mask = include_mask | data["_inst_name"].str.contains(keyword, na=False)
    data["score_career_focus_match"] = include_mask.astype(float)
else:
    data["score_career_focus_match"] = 0.5

if exclude_keywords:
    exclude_mask = pd.Series(False, index=data.index)
    for keyword in exclude_keywords:
        exclude_mask = exclude_mask | data["_inst_name"].str.contains(keyword, na=False)
    data = data.loc[~exclude_mask].copy()

if preferred_state != "Any" and "hd_STABBR" in data.columns:
    state_match = (data["hd_STABBR"].astype(str) == preferred_state)
    if enforce_state_filter:
        data = data.loc[state_match].copy()
        data["score_location_fit_personalized"] = 1.0
    else:
        data["score_location_fit_personalized"] = state_match.astype(float)
else:
    data["score_location_fit_personalized"] = data["score_location_fit_base"].fillna(0.5)

if "estimated_annual_cost" in data.columns:
    if require_known_cost:
        budget_ok = data["estimated_annual_cost"].notna() & (data["estimated_annual_cost"] <= max_budget)
    else:
        budget_ok = data["estimated_annual_cost"].isna() | (data["estimated_annual_cost"] <= max_budget)
    data = data.loc[budget_ok].copy()

if "score_cost_affordability" in data.columns:
    data["score_cost_affordability_user"] = data["score_cost_affordability"].fillna(0.2)
    if "estimated_annual_cost" in data.columns:
        missing_cost_mask = data["estimated_annual_cost"].isna()
        data.loc[missing_cost_mask, "score_cost_affordability_user"] = np.minimum(
            data.loc[missing_cost_mask, "score_cost_affordability_user"],
            0.2,
        )
else:
    data["score_cost_affordability_user"] = 0.2

if data.empty:
    st.warning("No institutions match your current state/career/budget filters. Relax one filter and try again.")
    st.stop()

data["score_composite_user"] = (
    data["score_academic_quality"].fillna(0.5) * weights["score_academic_quality"]
    + data["score_cost_affordability_user"].fillna(0.2) * weights["score_cost_affordability"]
    + ((data["score_career_proxy"].fillna(0.5) * 0.8) + (data["score_career_focus_match"].fillna(0.5) * 0.2))
    * weights["score_career_proxy"]
    + data["score_location_fit_personalized"].fillna(0.5) * weights["score_location_fit_base"]
    + data["score_safety_qol_proxy"].fillna(0.5) * weights["score_safety_qol_proxy"]
)

top = data.sort_values("score_composite_user", ascending=False).head(top_n).copy()
top.insert(0, "rank_user", np.arange(1, len(top) + 1))

display_cols = [
    "rank_user",
    "UNITID",
    "school_display_name",
    "hd_INSTNM",
    "hd_CITY",
    "hd_STABBR",
    "estimated_annual_cost",
    "score_academic_quality",
    "score_cost_affordability_user",
    "score_career_proxy",
    "score_career_focus_match",
    "score_location_fit_personalized",
    "score_safety_qol_proxy",
    "score_composite_user",
]
display_cols = [c for c in display_cols if c in top.columns]

st.subheader(f"Top {top_n} Matches")
st.dataframe(top[display_cols], use_container_width=True)

st.subheader("Why these results")
st.markdown(
    "- Scores are weighted by your slider settings and normalized to sum to 1."
    "\n- Unknown cost can be excluded (recommended) or penalized in scoring."
    "\n- Career preset keywords apply automatically unless you enter custom include keywords."
    "\n- Career/program keywords are soft-matched by default; enable strict filter only when needed."
    "\n- Career and safety use IPEDS proxies in this MVP (not direct placement/safety outcomes)."
    "\n- Use this as a decision-support prototype, not a definitive ranking."
)

with st.expander("Career keywords in effect"):
    st.write(effective_include_keywords)

with st.expander("Normalized Weights Used"):
    st.json(weights)
