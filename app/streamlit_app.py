from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.ar_model import AutoReg

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
    st.altair_chart(chart, use_container_width=True)


def autoreg_forecast(series: pd.Series, horizon: int = 4, max_lags: int = 4) -> pd.Series | None:
    clean = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    n_obs = len(clean)
    if n_obs < 6:
        return None

    lag = min(max_lags, n_obs - 2)
    if lag < 1:
        return None

    try:
        model = AutoReg(clean, lags=lag, old_names=False).fit()
        pred = model.predict(start=n_obs, end=n_obs + horizon - 1)
        return pd.Series(pred.values)
    except Exception:
        return None

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

# Price Trends: selection is constrained to institutions shown in the top table.
price_choice_df = top.copy()
price_choice_df["school_label"] = (
    price_choice_df.get("school_display_name", pd.Series("", index=price_choice_df.index))
    .fillna("")
    .astype(str)
)
blank_label_mask = price_choice_df["school_label"].str.strip() == ""
if "hd_INSTNM" in price_choice_df.columns:
    price_choice_df.loc[blank_label_mask, "school_label"] = price_choice_df.loc[blank_label_mask, "hd_INSTNM"].astype(str)

price_choice_df["selector_label"] = (
    "#"
    + price_choice_df["rank_user"].astype(int).astype(str)
    + " - "
    + price_choice_df["school_label"].astype(str)
    + " (UNITID: "
    + price_choice_df["UNITID"].astype(str)
    + ")"
)

st.subheader("Price Trends")
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

    price_trend_options = {
        "PT1": "Historical Tuition & Fees",
        "PT2": "Future Tuition & Fees (Est.)",
    }

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

panel_df = None
panel_error = None
if PANEL_COSTS_FILE.exists():
    try:
        panel_df = load_panel_costs(str(PANEL_COSTS_FILE))
    except Exception as exc:
        panel_error = str(exc)
else:
    panel_error = "Missing data/panel_costs.csv."

st.subheader(detail_box_title)
with st.container(border=True):
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
        if panel_error:
            st.warning(panel_error)
        else:
            selected_unitid = int(selected_price_row["UNITID"])
            school_panel = panel_df.loc[panel_df["unitid"].astype(int) == selected_unitid].copy()
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
            selected_unitid = int(selected_price_row["UNITID"])
            school_panel = panel_df.loc[panel_df["unitid"].astype(int) == selected_unitid].copy()
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
                    st.altair_chart(chart, use_container_width=True)

                    preview = pd.DataFrame(
                        {
                            "Year": future_years,
                            "Forecast_ISPrice": np.round(is_pred.values, 2),
                            "Forecast_OOSPrice": np.round(oos_pred.values, 2),
                        }
                    )
                    st.dataframe(preview, use_container_width=True)


st.subheader("Value-Added by Major")
with st.container(border=True):
    if not VALUE_ADDED_FILE.exists():
        st.warning("Missing data/value_added_by_major.csv. Add this file to enable major-level value-added charts.")
    else:
        try:
            va_df = load_value_added(str(VALUE_ADDED_FILE))
            va_df["CAMPUS_LABEL"] = va_df["INSTNM"] + " (UNITID: " + va_df["UNITID"].astype(str) + ")"

            campus_labels = sorted(va_df["CAMPUS_LABEL"].dropna().unique().tolist())
            if not campus_labels:
                st.warning("No campus rows available in value-added dataset.")
            else:
                selected_unitid = int(selected_price_row["UNITID"])
                default_label = None
                school_match = va_df.loc[va_df["UNITID"] == selected_unitid, "CAMPUS_LABEL"]
                if not school_match.empty:
                    default_label = str(school_match.iloc[0])

                default_index = campus_labels.index(default_label) if default_label in campus_labels else 0

                selected_campus = st.selectbox(
                    "Select Campus",
                    options=campus_labels,
                    index=default_index,
                    key="va_selected_campus",
                )

                selected_campus_unitid = int(float(selected_campus.split("UNITID: ")[1].replace(")", "")))
                school_va = va_df.loc[va_df["UNITID"] == selected_campus_unitid].copy()

                field_options = sorted([x for x in school_va["CIPDESC"].dropna().unique().tolist() if str(x).strip()])
                if not field_options:
                    st.warning("No fields of study available for selected campus.")
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
                        st.warning("No data available for this campus/field/metric selection.")
                    else:
                        row = filtered.iloc[0]
                        actual = pd.to_numeric(row.get("earnings_actual"), errors="coerce")
                        pred = pd.to_numeric(row.get("earnings_pred"), errors="coerce")
                        va = pd.to_numeric(row.get("earnings_va"), errors="coerce")
                        zscore = pd.to_numeric(row.get("earnings_va_z"), errors="coerce")

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
                            st.altair_chart(bar, use_container_width=True)

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

st.subheader("Selection Map")
with st.container(border=True):
    st.caption("Scaffold block for add/delete iterations. Remove anytime.")
    st.write(
        {
            "selected_unitid": str(selected_price_row["UNITID"]),
            "selected_school": str(selected_price_row["school_label"]),
            "price_trends_code": trend_mode_code,
            "price_trends_label": price_trend_options[trend_mode_code],
            "detail_box_label": detail_box_title,
            "historical_code": historical_mode_code if historical_mode_code is not None else "N/A (requires PT1)",
            "historical_label": historical_options[historical_mode_code] if historical_mode_code is not None else "N/A (requires PT1)",
        }
    )

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
