from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SCORES_FILE = OUT / "recommendation_base_scores.csv"
PANEL_COSTS_FILE = ROOT / "data" / "panel_costs.csv"
VALUE_ADDED_FILE = ROOT / "data" / "value_added_by_major.csv"


@st.cache_data
def load_panel_costs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    needed = {"Year", "unitid", "ISPrice", "OOSPrice"}
    if not needed.issubset(df.columns):
        missing = sorted(needed - set(df.columns))
        raise ValueError(f"panel_costs.csv is missing required columns: {missing}")
    df = df[["Year", "unitid", "ISPrice", "OOSPrice"]].copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["unitid"] = pd.to_numeric(df["unitid"], errors="coerce")
    df["ISPrice"] = pd.to_numeric(df["ISPrice"], errors="coerce")
    df["OOSPrice"] = pd.to_numeric(df["OOSPrice"], errors="coerce")
    return df.dropna(subset=["Year", "unitid"]).copy()


@st.cache_data
def load_value_added(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    needed = {
        "UNITID",
        "CIPDESC",
        "INSTNM",
        "EARNINGS_METRIC",
        "earnings_actual",
        "earnings_pred",
        "earnings_va",
        "earnings_va_z",
    }
    if not needed.issubset(df.columns):
        missing = sorted(needed - set(df.columns))
        raise ValueError(f"value_added_by_major.csv is missing required columns: {missing}")

    df = df.copy()
    df["UNITID"] = pd.to_numeric(df["UNITID"], errors="coerce")
    df = df.dropna(subset=["UNITID"]).copy()
    df["UNITID"] = df["UNITID"].astype(int)
    df["CIPDESC"] = df["CIPDESC"].astype(str).str.strip()
    df["INSTNM"] = df["INSTNM"].astype(str).str.strip()
    return df


def render_line_chart(df: pd.DataFrame, title: str) -> None:
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Year:Q", title="Year"),
            y=alt.Y("Value:Q"),
            color=alt.Color("Series:N", title="Series"),
            tooltip=["Year:Q", "Series:N", alt.Tooltip("Value:Q", format=",.2f")],
        )
        .properties(height=320, title=title)
    )
    st.altair_chart(chart, width="stretch")


def autoreg_forecast(series: pd.Series, horizon: int = 4, max_lags: int = 4) -> pd.Series | None:
    clean = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    n_obs = len(clean)
    if n_obs < 6:
        return None

    lag = min(max_lags, n_obs - 2)
    if lag < 1:
        return None

    try:
        from statsmodels.tsa.ar_model import AutoReg  # pyright: ignore[reportMissingImports]

        model = AutoReg(clean, lags=lag, old_names=False).fit()
        pred = model.predict(start=n_obs, end=n_obs + horizon - 1)
        return pd.Series(pred.values)
    except Exception:
        return None


def safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def build_school_catalog(df: pd.DataFrame) -> pd.DataFrame:
    catalog = df.copy()
    catalog["UNITID"] = pd.to_numeric(catalog["UNITID"], errors="coerce")
    catalog = catalog.dropna(subset=["UNITID"]).copy()
    catalog["UNITID"] = catalog["UNITID"].astype(int)

    if "school_display_name" in catalog.columns:
        primary_name = catalog["school_display_name"].astype(str).fillna("")
    else:
        primary_name = pd.Series("", index=catalog.index)

    fallback_name = catalog.get("hd_INSTNM", pd.Series("", index=catalog.index)).astype(str).fillna("")
    school_name = primary_name.where(primary_name.str.strip() != "", fallback_name)

    city = catalog.get("hd_CITY", pd.Series("", index=catalog.index)).astype(str).fillna("")
    state = catalog.get("hd_STABBR", pd.Series("", index=catalog.index)).astype(str).fillna("")
    city_state = city.str.strip() + ", " + state.str.strip()
    city_state = city_state.str.strip(" ,")

    out = pd.DataFrame(
        {
            "UNITID": catalog["UNITID"],
            "school_name": school_name,
            "city_state": city_state,
        }
    ).drop_duplicates(subset=["UNITID"])

    out["school_label"] = (
        out["school_name"].astype(str)
        + " ("
        + out["city_state"].astype(str)
        + ")"
        + " (UNITID: "
        + out["UNITID"].astype(str)
        + ")"
    )
    return out.sort_values("school_name").reset_index(drop=True)


def extract_unitid_from_label(selection: str) -> int | None:
    token = "UNITID: "
    if token not in selection:
        return None
    try:
        raw = selection.split(token, 1)[1].replace(")", "").strip()
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def resolve_school_unitid(selection: str | None, catalog: pd.DataFrame) -> int | None:
    if not selection or selection == "None":
        return None

    parsed = extract_unitid_from_label(selection)
    if parsed is not None:
        return parsed

    exact_label = catalog.loc[catalog["school_label"].str.lower() == selection.lower(), "UNITID"]
    if not exact_label.empty:
        return int(exact_label.iloc[0])

    exact_name = catalog.loc[catalog["school_name"].str.lower() == selection.lower(), "UNITID"]
    if not exact_name.empty:
        return int(exact_name.iloc[0])

    contains_name = catalog.loc[catalog["school_name"].str.lower().str.contains(selection.lower(), na=False), "UNITID"]
    if not contains_name.empty:
        return int(contains_name.iloc[0])

    return None


def build_complete_school_whitelist(
    base_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    va_df: pd.DataFrame,
    min_panel_points: int = 6,
) -> list[int]:
    base_ids = set(pd.to_numeric(base_df["UNITID"], errors="coerce").dropna().astype(int))

    panel_data = panel_df.copy()
    panel_data["unitid"] = pd.to_numeric(panel_data["unitid"], errors="coerce")
    panel_data["Year"] = pd.to_numeric(panel_data["Year"], errors="coerce")
    panel_data["ISPrice"] = pd.to_numeric(panel_data["ISPrice"], errors="coerce")
    panel_data["OOSPrice"] = pd.to_numeric(panel_data["OOSPrice"], errors="coerce")
    panel_data = panel_data.dropna(subset=["unitid", "Year"]).copy()
    panel_data["unitid"] = panel_data["unitid"].astype(int)

    panel_valid = panel_data.dropna(subset=["ISPrice", "OOSPrice"]).copy()
    panel_counts = panel_valid.groupby("unitid")[["ISPrice", "OOSPrice"]].count()
    panel_ids = set(
        panel_counts.loc[
            (panel_counts["ISPrice"] >= min_panel_points) & (panel_counts["OOSPrice"] >= min_panel_points)
        ].index.astype(int)
    )

    va_ids = set(pd.to_numeric(va_df["UNITID"], errors="coerce").dropna().astype(int))
    return sorted(base_ids & panel_ids & va_ids)


HELP_SECTIONS: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "HOV",
        "App Overview",
        [
            ("APP", "College Match MVP screen that ranks schools from your current preference settings."),
            ("HLP", "This popup help guide. Codes in parentheses are internal references for faster future edits."),
        ],
    ),
    (
        "SPR",
        "Preferences (Sidebar)",
        [
            ("BGT", "Max Annual Budget input. Upper-cost schools are filtered out from results."),
            ("CKC", "Require known annual cost checkbox. If on, schools missing cost data are excluded."),
            ("TOP", "How many results slider. Controls the row count in the Top Matches table."),
            ("CFP", "Career focus preset dropdown. Adds keyword logic for program fit."),
            ("SKF", "Strict career filter checkbox. If on, only schools matching include keywords remain."),
            ("IKW", "Include keywords input. Optional comma-separated school-name keywords for career fit."),
            ("EKW", "Exclude keywords input. Optional comma-separated school-name keywords to remove."),
            ("WAC", "Academic Quality weight slider."),
            ("WCO", "Cost and Aid weight slider."),
            ("WCR", "Career Outcomes (Proxy) weight slider."),
            ("WLO", "Location Fit weight slider."),
            ("WSF", "Safety and Quality-of-Life (Proxy) weight slider."),
        ],
    ),
    (
        "STM",
        "Top Matches",
        [
            ("TBT", "Top X Matches table showing ranked schools and component scores."),
            ("PST", "Select another school dropdown that extends the Top Matches option list when valid and not already present (for example, Top 10 + 1)."),
            ("RKU", "User rank column generated from personalized composite score."),
            ("SCU", "Personalized composite score column used to sort final recommendations."),
        ],
    ),
    (
        "SPT",
        "Price Trends",
        [
            ("PTD", "Selected School dropdown list uses the combined candidate list from (TBT) plus valid (PST) extension entries."),
            ("PTV", "View control toggles between historical and forecast mode."),
            ("PT1", "Historical Tuition and Fees; populates the Historical section with (HS1) Price vs. Median selected (default)."),
            ("PT2", "Future Tuition and Fees estimate; populates forecast graph and table from AutoReg projections."),
            ("PSL", "School label display reflecting the currently selected school option from (PTD)."),
            ("PUN", "Linked UNITID caption for the currently selected school."),
        ],
    ),
    (
        "SHS",
        "Historical",
        [
            ("HS1", "Price vs. Median control requests and displays both (GP1) In-State Price vs. Median graph and (GP2) Out-of-State vs. Median graph, based on (PTD)."),
            ("HS2", "Price Difference from Market control displays (GP3) In-State Dollar Delta and (GP4) Out-of-State Dollar Delta graphs."),
            ("HS3", "Year-over-Year Price Change vs. Market control displays (GP5) In-State YoY vs. Median and (GP6) Out-of-State YoY vs. Median graphs."),
            ("GP1", "In-State Price vs. Median line graph."),
            ("GP2", "Out-of-State Price vs. Median line graph."),
            ("GP3", "In-State dollar-difference-from-median graph."),
            ("GP4", "Out-of-State dollar-difference-from-median graph."),
            ("GP5", "In-State YoY percent-change vs. median graph."),
            ("GP6", "Out-of-State YoY percent-change vs. median graph."),
            ("E01", "No panel data available warning for selected UNITID."),
        ],
    ),
    (
        "SFT",
        "Future Tuition+Fees Estimate",
        [
            ("FP1", "Combined Actual and 4-Year Forecast Tuition chart."),
            ("FP2", "Forecast preview table with Year, Forecast_ISPrice, and Forecast_OOSPrice."),
            ("FP3", "Overlap-detection note shown when in-state and out-of-state are identical."),
            ("FP4", "Split In-State forecast panel shown when overlap is detected."),
            ("FP5", "Split Out-of-State forecast panel shown when overlap is detected."),
            ("E02", "Not enough historical points warning for forecast model."),
        ],
    ),
    (
        "SVA",
        "Value-Added by Major",
        [
            ("VAD", "Campus selector is temporarily suppressed in the UI."),
            ("VSO", "Value-Added source school selector uses the same combined candidate list as (PTD): (TBT) rows plus valid (PST) extension entries."),
            ("VAF", "Select Field of Study dropdown, filtered by the active source school."),
            ("VAM", "Select Earnings Metric dropdown (1-year, 4-year, 5-year)."),
            ("VMC", "Summary metric cards for Actual, Predicted, Value-Added, and Z-Score."),
            ("VBG", "Actual vs. Predicted earnings bar chart."),
            ("VIN", "Interpretation message showing above/below expected earnings."),
            ("E03", "Value-added file missing warning."),
            ("E04", "No available campus/field/metric rows warning."),
        ],
    ),
    (
        "SSM",
        "Selection Map",
        [
            ("MAP", "Debug scaffold map of currently selected school, section, and view codes."),
        ],
    ),
    (
        "SRS",
        "Why These Results",
        [
            ("RSN", "Explanation block describing scoring behavior and prototype caveats."),
            ("EXP", "Career keywords expander and normalized weights expander for transparency."),
        ],
    ),
]


def render_help_popup() -> None:
    with st.popover("Help", use_container_width=False):
        st.markdown("### User Guide")
        st.caption(
            "Each item includes a stable reference code in parentheses. "
            "Codes are for maintenance and only shown in this Help popup. "
            "Several UNITID-to-school mapping paths are supported, so outputs may vary slightly when only part of the dependency chain resolves."
        )

        for section_code, section_title, items in HELP_SECTIONS:
            st.markdown(f"**({section_code}) {section_title}**")
            for item_code, description in items:
                st.markdown(f"- ({item_code}) {description}")

st.set_page_config(page_title="College Match MVP", layout="wide")
title_col, help_col = st.columns([0.88, 0.12])
with title_col:
    st.title("College Match MVP")
    st.caption("Top-N recommendations from IPEDS-based component scoring")
with help_col:
    render_help_popup()

if not SCORES_FILE.exists():
    st.error("Missing outputs/recommendation_base_scores.csv. Run scripts/07_build_recommendation_scores.py first.")
    st.stop()

if not PANEL_COSTS_FILE.exists():
    st.error("Missing data/panel_costs.csv. Run the data build pipeline first.")
    st.stop()

if not VALUE_ADDED_FILE.exists():
    st.error("Missing data/value_added_by_major.csv. Run the data build pipeline first.")
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

panel_df = load_panel_costs(str(PANEL_COSTS_FILE))
va_df = load_value_added(str(VALUE_ADDED_FILE))

complete_school_unitids = build_complete_school_whitelist(base, panel_df, va_df)
if not complete_school_unitids:
    st.error("No schools have complete coverage across Top Matches, Price Trends, and Value-Added data.")
    st.stop()

complete_school_unitid_set = set(complete_school_unitids)
complete_school_mask = base["UNITID"].astype("Int64").isin(complete_school_unitid_set)
base = base.loc[complete_school_mask].copy()

school_catalog = build_school_catalog(base)

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
st.dataframe(top[display_cols], width="stretch")

selected_override_label = st.selectbox(
    "Select another school",
    options=["None"] + school_catalog["school_label"].tolist(),
    index=0,
    key="select_another_school",
    accept_new_options=False,
    help="Extends the Top Matches candidate list for both Price Trends and Value-Added when valid.",
)
selected_override_unitid = resolve_school_unitid(selected_override_label, school_catalog)
if selected_override_label != "None" and selected_override_unitid is None:
    st.warning("Could not resolve the typed school to a UNITID. Please choose a valid entry from the list.")

# Price Trends and Value-Added share a combined candidate list: TBT rows plus PST (if valid and not already in TBT).
price_choice_df = top.copy()
price_choice_df["school_label"] = (
    price_choice_df.get("school_display_name", pd.Series("", index=price_choice_df.index))
    .fillna("")
    .astype(str)
)
blank_label_mask = price_choice_df["school_label"].str.strip() == ""
if "hd_INSTNM" in price_choice_df.columns:
    price_choice_df.loc[blank_label_mask, "school_label"] = price_choice_df.loc[blank_label_mask, "hd_INSTNM"].astype(str)

if selected_override_unitid is not None and int(selected_override_unitid) not in set(price_choice_df["UNITID"].astype(int).tolist()):
    extra_row = school_catalog.loc[school_catalog["UNITID"].astype(int) == int(selected_override_unitid)]
    if not extra_row.empty:
        extra_label = str(extra_row.iloc[0]["school_name"]).strip()
        if not extra_label:
            extra_label = str(extra_row.iloc[0]["school_label"])
    else:
        extra_label = str(selected_override_label)

    appended = pd.DataFrame(
        {
            "rank_user": [int(len(price_choice_df) + 1)],
            "UNITID": [int(selected_override_unitid)],
            "school_label": [extra_label],
        }
    )
    price_choice_df = pd.concat([price_choice_df, appended], ignore_index=True)

price_choice_df["UNITID"] = pd.to_numeric(price_choice_df["UNITID"], errors="coerce").astype("Int64")
price_choice_df = price_choice_df.dropna(subset=["UNITID"]).copy()
price_choice_df["UNITID"] = price_choice_df["UNITID"].astype(int)

if price_choice_df.empty:
    st.error(
        "Whitelist invariant violated: no selectable schools remain after applying filters. "
        "Adjust filters or rebuild data so at least one whitelisted UNITID survives."
    )
    st.stop()

price_choice_df["rank_user"] = np.arange(1, len(price_choice_df) + 1)

price_choice_df["selector_label"] = (
    "#"
    + price_choice_df["rank_user"].astype(int).astype(str)
    + " - "
    + price_choice_df["school_label"].astype(str)
    + " (UNITID: "
    + price_choice_df["UNITID"].astype(str)
    + ")"
)

# Keep PTD and VAD in sync with PST updates.
default_selector_label = str(price_choice_df.iloc[0]["selector_label"]) if not price_choice_df.empty else ""
pst_selector_label = default_selector_label
if selected_override_unitid is not None and not price_choice_df.empty:
    pst_match = price_choice_df.index[price_choice_df["UNITID"] == int(selected_override_unitid)].tolist()
    if pst_match:
        pst_selector_label = str(price_choice_df.iloc[pst_match[0]]["selector_label"])

prev_pst_label = st.session_state.get("_pst_last_seen_label")
pst_changed = prev_pst_label != selected_override_label

if pst_changed and default_selector_label:
    propagated_label = pst_selector_label if selected_override_unitid is not None else default_selector_label
    st.session_state["price_trends_school_selector"] = propagated_label
    st.session_state["va_source_school"] = propagated_label

st.session_state["_pst_last_seen_label"] = selected_override_label

if "price_trends_school_selector" in st.session_state and st.session_state["price_trends_school_selector"] not in set(price_choice_df["selector_label"].tolist()):
    st.session_state["price_trends_school_selector"] = default_selector_label
if "va_source_school" in st.session_state and st.session_state["va_source_school"] not in set(price_choice_df["selector_label"].tolist()):
    st.session_state["va_source_school"] = default_selector_label

price_trend_options = {
    "PT1": "Historical Tuition & Fees",
    "PT2": "Future Tuition & Fees (Est.)",
}
selected_price_row = price_choice_df.iloc[0]
trend_mode_code = "PT1"

with st.expander("Price Trends", expanded=True):
    with st.container(border=True):
        school_option = st.selectbox(
            "Selected school (from Top Matches)",
            options=price_choice_df["selector_label"].tolist(),
            index=0,
            key="price_trends_school_selector",
        )

        selected_price_row = price_choice_df.loc[price_choice_df["selector_label"] == school_option].iloc[0]
        st.markdown(f"**School:** {selected_price_row['school_label']}")
        st.caption(f"Linked UNITID: {selected_price_row['UNITID']}")

        trend_mode_code = st.radio(
            "View",
            options=list(price_trend_options.keys()),
            format_func=lambda code: f"{code} - {price_trend_options[code]}",
            horizontal=True,
            key="price_trends_mode",
        )

        if trend_mode_code == "PT1":
            st.info("PT1 selected. Historical Tuition & Fees is active.")
        else:
            st.info("PT2 selected. Future Tuition & Fees (Est.) is active with AR forecasts.")
        historical_mode_code = None
        detail_box_title = "Historical" if trend_mode_code == "PT1" else "Future Tuition+Fees Estimate"

        panel_error = None

        with st.expander(detail_box_title, expanded=True):
            st.markdown(f"**School:** {selected_price_row['school_label']}")
            st.caption(f"Linked UNITID: {selected_price_row['UNITID']}")

            historical_options = {
                "HS1": "Price vs. Median",
                "HS2": "Price Difference from Market",
                "HS3": "Year over Year Price Change % vs. Market",
            }

            if trend_mode_code == "PT1":
                historical_mode_code = st.radio(
                    "Historical View",
                    options=list(historical_options.keys()),
                    format_func=lambda code: f"{code} - {historical_options[code]}",
                    horizontal=True,
                    key="historical_mode",
                )
                if panel_df is None:
                    st.warning(panel_error or "Missing data/panel_costs.csv.")
                elif panel_error:
                    st.warning(panel_error)
                else:
                    panel_data = panel_df
                    assert panel_data is not None
                    selected_unitid = int(selected_price_row["UNITID"])
                    school_panel = panel_data.loc[panel_data["unitid"].astype(int) == selected_unitid].copy()
                    school_panel = school_panel.sort_values("Year").reset_index(drop=True)

                    if school_panel.empty:
                        st.warning("No panel cost rows found for selected UNITID.")
                    else:
                        median_by_year = (
                            panel_df.groupby("Year", as_index=False)[["ISPrice", "OOSPrice"]]
                            .median()
                            .rename(columns={"ISPrice": "Median_ISPrice", "OOSPrice": "Median_OOSPrice"})
                        )
                        comp = school_panel.merge(median_by_year, on="Year", how="left")
                        comp["IS_Dollar_Diff"] = comp["ISPrice"] - comp["Median_ISPrice"]
                        comp["OOS_Dollar_Diff"] = comp["OOSPrice"] - comp["Median_OOSPrice"]
                        comp["IS_YOY"] = comp["ISPrice"].pct_change() * 100
                        comp["Median_IS_YOY"] = comp["Median_ISPrice"].pct_change() * 100
                        comp["OOS_YOY"] = comp["OOSPrice"].pct_change() * 100
                        comp["Median_OOS_YOY"] = comp["Median_OOSPrice"].pct_change() * 100

                        if historical_mode_code == "HS1":
                            is_hist = pd.DataFrame(
                                {
                                    "Year": comp["Year"],
                                    "Series": "School In-State",
                                    "Value": comp["ISPrice"],
                                }
                            )
                            is_med = pd.DataFrame(
                                {
                                    "Year": comp["Year"],
                                    "Series": "Median In-State",
                                    "Value": comp["Median_ISPrice"],
                                }
                            )
                            oos_hist = pd.DataFrame(
                                {
                                    "Year": comp["Year"],
                                    "Series": "School Out-of-State",
                                    "Value": comp["OOSPrice"],
                                }
                            )
                            oos_med = pd.DataFrame(
                                {
                                    "Year": comp["Year"],
                                    "Series": "Median Out-of-State",
                                    "Value": comp["Median_OOSPrice"],
                                }
                            )
                            render_line_chart(pd.concat([is_hist, is_med], ignore_index=True), "In-State Price vs Median")
                            render_line_chart(pd.concat([oos_hist, oos_med], ignore_index=True), "Out-of-State Price vs Median")

                        elif historical_mode_code == "HS2":
                            diff_is = pd.DataFrame(
                                {
                                    "Year": comp["Year"],
                                    "Series": "In-State Price Delta",
                                    "Value": comp["IS_Dollar_Diff"],
                                }
                            )
                            diff_oos = pd.DataFrame(
                                {
                                    "Year": comp["Year"],
                                    "Series": "Out-of-State Price Delta",
                                    "Value": comp["OOS_Dollar_Diff"],
                                }
                            )
                            render_line_chart(diff_is, "In-State Dollar Difference from Median")
                            render_line_chart(diff_oos, "Out-of-State Dollar Difference from Median")

                        else:
                            yoy_is = pd.DataFrame(
                                {
                                    "Year": comp["Year"],
                                    "Series": "School In-State YoY %",
                                    "Value": comp["IS_YOY"],
                                }
                            )
                            yoy_is_med = pd.DataFrame(
                                {
                                    "Year": comp["Year"],
                                    "Series": "Median In-State YoY %",
                                    "Value": comp["Median_IS_YOY"],
                                }
                            )
                            yoy_oos = pd.DataFrame(
                                {
                                    "Year": comp["Year"],
                                    "Series": "School Out-of-State YoY %",
                                    "Value": comp["OOS_YOY"],
                                }
                            )
                            yoy_oos_med = pd.DataFrame(
                                {
                                    "Year": comp["Year"],
                                    "Series": "Median Out-of-State YoY %",
                                    "Value": comp["Median_OOS_YOY"],
                                }
                            )
                            render_line_chart(pd.concat([yoy_is, yoy_is_med], ignore_index=True), "In-State YoY Change vs Median")
                            render_line_chart(pd.concat([yoy_oos, yoy_oos_med], ignore_index=True), "Out-of-State YoY Change vs Median")
            else:
                if panel_error:
                    st.warning(panel_error)
                else:
                    panel_data = panel_df
                    assert panel_data is not None
                    selected_unitid = int(selected_price_row["UNITID"])
                    school_panel = panel_data.loc[panel_data["unitid"].astype(int) == selected_unitid].copy()
                    school_panel = school_panel.sort_values("Year").reset_index(drop=True)

                    if school_panel.empty:
                        st.warning("No panel cost rows found for selected UNITID.")
                    else:
                        is_pred = autoreg_forecast(school_panel["ISPrice"], horizon=4)
                        oos_pred = autoreg_forecast(school_panel["OOSPrice"], horizon=4)

                        if is_pred is None or oos_pred is None:
                            st.warning("Not enough historical points to produce AutoReg forecast for this school.")
                        else:
                            last_year = int(pd.to_numeric(school_panel["Year"], errors="coerce").max())
                            future_years = pd.Series(range(last_year + 1, last_year + 5), name="Year")

                            actual_is = pd.DataFrame(
                                {
                                    "Year": school_panel["Year"],
                                    "Value": school_panel["ISPrice"],
                                    "Tuition": "In-State",
                                    "Status": "Actual",
                                }
                            )
                            actual_oos = pd.DataFrame(
                                {
                                    "Year": school_panel["Year"],
                                    "Value": school_panel["OOSPrice"],
                                    "Tuition": "Out-of-State",
                                    "Status": "Actual",
                                }
                            )
                            fc_is = pd.DataFrame(
                                {
                                    "Year": future_years,
                                    "Value": is_pred.values,
                                    "Tuition": "In-State",
                                    "Status": "Forecast",
                                }
                            )
                            fc_oos = pd.DataFrame(
                                {
                                    "Year": future_years,
                                    "Value": oos_pred.values,
                                    "Tuition": "Out-of-State",
                                    "Status": "Forecast",
                                }
                            )
                            forecast_df = pd.concat([actual_is, actual_oos, fc_is, fc_oos], ignore_index=True)

                            same_history = np.allclose(
                                pd.to_numeric(school_panel["ISPrice"], errors="coerce").to_numpy(dtype=np.float64),
                                pd.to_numeric(school_panel["OOSPrice"], errors="coerce").to_numpy(dtype=np.float64),
                                equal_nan=True,
                            )
                            same_forecast = np.allclose(
                                is_pred.to_numpy(dtype=np.float64),
                                oos_pred.to_numpy(dtype=np.float64),
                                equal_nan=True,
                            )

                            if same_history and same_forecast:
                                st.info(
                                    "For this school, in-state and out-of-state values are identical in the source data for history and forecast. "
                                    "A combined chart would overlap both lines exactly, so the app shows separate panels below."
                                )

                                in_state_df = pd.concat([actual_is, fc_is], ignore_index=True)
                                out_state_df = pd.concat([actual_oos, fc_oos], ignore_index=True)

                                c_left, c_right = st.columns(2)
                                with c_left:
                                    in_chart = (
                                        alt.Chart(in_state_df)
                                        .mark_line(point=True)
                                        .encode(
                                            x=alt.X("Year:Q", title="Year"),
                                            y=alt.Y("Value:Q", title="Tuition / Fees"),
                                            color=alt.Color("Status:N", title="Series"),
                                            tooltip=["Year:Q", "Status:N", alt.Tooltip("Value:Q", format=",.2f")],
                                        )
                                        .properties(height=320, title="In-State: Actual and 4-Year Forecast")
                                    )
                                    st.altair_chart(in_chart, width="stretch")

                                with c_right:
                                    out_chart = (
                                        alt.Chart(out_state_df)
                                        .mark_line(point=True)
                                        .encode(
                                            x=alt.X("Year:Q", title="Year"),
                                            y=alt.Y("Value:Q", title="Tuition / Fees"),
                                            color=alt.Color("Status:N", title="Series"),
                                            tooltip=["Year:Q", "Status:N", alt.Tooltip("Value:Q", format=",.2f")],
                                        )
                                        .properties(height=320, title="Out-of-State: Actual and 4-Year Forecast")
                                    )
                                    st.altair_chart(out_chart, width="stretch")
                            else:
                                chart = (
                                    alt.Chart(forecast_df)
                                    .mark_line(point=True)
                                    .encode(
                                        x=alt.X("Year:Q", title="Year"),
                                        y=alt.Y("Value:Q", title="Tuition / Fees"),
                                        color=alt.Color("Tuition:N", title="Tuition Type"),
                                        strokeDash=alt.StrokeDash("Status:N", title="Series"),
                                        tooltip=[
                                            "Year:Q",
                                            "Tuition:N",
                                            "Status:N",
                                            alt.Tooltip("Value:Q", format=",.2f"),
                                        ],
                                    )
                                    .properties(height=360, title="Actual and 4-Year Forecast Tuition")
                                )
                                st.altair_chart(chart, width="stretch")

                            preview = pd.DataFrame(
                                {
                                    "Year": future_years,
                                    "Forecast_ISPrice": np.round(is_pred.to_numpy(dtype=np.float64), 2),
                                    "Forecast_OOSPrice": np.round(oos_pred.to_numpy(dtype=np.float64), 2),
                                }
                            )
                            st.dataframe(preview, width="stretch")


with st.expander("Value-Added by Major", expanded=True):
  with st.container(border=True):
    if not VALUE_ADDED_FILE.exists():
        st.warning("Missing data/value_added_by_major.csv. Add this file to enable major-level value-added charts.")
    else:
        try:
            va_source_default_unitid = int(selected_price_row["UNITID"])
            if selected_override_unitid is not None and int(selected_override_unitid) in set(price_choice_df["UNITID"].astype(int).tolist()):
                va_source_default_unitid = int(selected_override_unitid)

            va_source_labels = price_choice_df["selector_label"].tolist()
            if not va_source_labels:
                st.error("Whitelist invariant violated: Value-Added source list is empty.")
                st.stop()
            else:
                va_default_match_pos = np.where(price_choice_df["UNITID"].to_numpy(dtype=np.int64) == va_source_default_unitid)[0]
                if len(va_default_match_pos) == 0:
                    st.error(
                        "Whitelist invariant violated: selected UNITID not present in Value-Added source options."
                    )
                    st.stop()
                va_source_index = int(va_default_match_pos[0])

                va_source_choice = st.selectbox(
                    "Source School (from Top Matches + PST)",
                    options=va_source_labels,
                    index=va_source_index,
                    key="va_source_school",
                )
                selected_sva_row = price_choice_df.loc[price_choice_df["selector_label"] == va_source_choice].iloc[0]
                selected_sva_unitid = int(selected_sva_row["UNITID"])
                school_va = va_df.loc[va_df["UNITID"] == selected_sva_unitid].copy()

                if school_va.empty:
                    st.warning("No value-added rows available for the currently selected school.")
                else:
                    selected_school_name = str(school_va["INSTNM"].iloc[0])
                    st.caption(f"Source school: {selected_school_name} (UNITID: {selected_sva_unitid})")

                    field_options = sorted([x for x in school_va["CIPDESC"].dropna().unique().tolist() if str(x).strip()])
                    if not field_options:
                        st.warning("No fields of study available for selected school.")
                    else:
                        selected_field = st.selectbox(
                            "Select Field of Study",
                            options=field_options,
                            index=0,
                            key="va_selected_field",
                        )

                        metric_map = {
                            "EARN_MDN_1YR": "1-Year Median Earnings",
                            "EARN_MDN_4YR": "4-Year Median Earnings",
                            "EARN_MDN_5YR": "5-Year Median Earnings",
                        }
                        selected_metric_label = st.selectbox(
                            "Select Earnings Metric",
                            options=list(metric_map.values()),
                            index=0,
                            key="va_selected_metric",
                        )
                        selected_metric = [k for k, v in metric_map.items() if v == selected_metric_label][0]

                        filtered = school_va.loc[
                            (school_va["CIPDESC"] == selected_field)
                            & (school_va["EARNINGS_METRIC"] == selected_metric)
                        ].copy()

                        if filtered.empty:
                            st.warning("No data available for this field/metric selection.")
                        else:
                            row = filtered.iloc[0]
                            actual = safe_float(row["earnings_actual"])
                            pred = safe_float(row["earnings_pred"])
                            va = safe_float(row["earnings_va"])
                            zscore = safe_float(row["earnings_va_z"])

                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Actual Earnings", f"${actual:,.0f}" if pd.notna(actual) else "N/A")
                            c2.metric("Predicted Earnings", f"${pred:,.0f}" if pd.notna(pred) else "N/A")
                            c3.metric("Value-Added", f"${va:,.0f}" if pd.notna(va) else "N/A")
                            c4.metric("VA Z-Score", f"{zscore:.2f}" if pd.notna(zscore) else "N/A")

                            compare_df = pd.DataFrame(
                                {
                                    "Type": ["Actual Earnings", "Predicted Earnings"],
                                    "Value": [actual, pred],
                                }
                            )
                            compare_df = compare_df.dropna(subset=["Value"])

                            if not compare_df.empty:
                                bar = (
                                    alt.Chart(compare_df)
                                    .mark_bar()
                                    .encode(
                                        x=alt.X("Type:N", sort=None),
                                        y=alt.Y("Value:Q", title="Earnings"),
                                        color=alt.Color("Type:N", title="Metric"),
                                        tooltip=["Type:N", alt.Tooltip("Value:Q", format=",.0f")],
                                    )
                                    .properties(height=320)
                                )
                                st.altair_chart(bar, width="stretch")

                            if pd.notna(va) and pd.notna(zscore):
                                if va > 0:
                                    st.success(
                                        f"Graduates in {selected_field} are above model expectation by ${va:,.0f} (z={zscore:.2f})."
                                    )
                                else:
                                    st.error(
                                        f"Graduates in {selected_field} are below model expectation by ${abs(va):,.0f} (z={zscore:.2f})."
                                    )
        except Exception as exc:
            st.warning(f"Could not load value-added dashboard data: {exc}")

with st.expander("Selection Map", expanded=False):
    with st.container(border=True):
        st.caption("Scaffold block for add/delete iterations. Several UNITID-to-school mapping paths are supported, so the observed mapping can differ when only some dependencies resolve.")
        st.write(
            {
                "selected_unitid": str(selected_price_row["UNITID"]),
                "selected_school": str(selected_price_row["school_label"]),
                "price_trends_code": trend_mode_code,
                "price_trends_label": price_trend_options[trend_mode_code],
                "detail_box_label": detail_box_title,
                "historical_code": historical_mode_code if historical_mode_code is not None else "N/A (requires PT1)",
                "historical_label": historical_options[historical_mode_code] if historical_mode_code is not None else "N/A (requires PT1)",
                "pst_override": selected_override_label,
                "pst_unitid": str(selected_override_unitid) if selected_override_unitid is not None else "N/A",
                "tbt_plus_pst_options": str(len(price_choice_df)),
            }
        )

with st.expander("Why these results", expanded=False):
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
