import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
data_path = ROOT / "value_added_by_major.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(data_path, low_memory=False)

    # Clean identifiers to prevent hidden majors or mismatches
    df["UNITID"] = df["UNITID"].astype(float).astype(int)
    df["CIPDESC"] = df["CIPDESC"].astype(str).str.strip()
    df["INSTNM"] = df["INSTNM"].astype(str).str.strip()

    return df

df = load_data()

# ---------------------------------------------------------
# Sidebar Filters (Revised: UNITID → Majors)
# ---------------------------------------------------------
st.sidebar.header("Filters")

# Create a combined label for campus selection
df["CAMPUS_LABEL"] = df["INSTNM"] + " (UNITID: " + df["UNITID"].astype(str) + ")"

# School/Campus Filter FIRST
campuses = sorted(df["CAMPUS_LABEL"].dropna().unique())
selected_campus = st.sidebar.selectbox("Select Campus", campuses)

# Extract UNITID from the label
unitid_str = selected_campus.split("UNITID: ")[1].replace(")", "")
selected_unitid = int(float(unitid_str))

# Filter dataset by selected UNITID
df_school = df[df["UNITID"] == selected_unitid]

# Field of Study Filter (dependent on campus)
fields = sorted(df_school["CIPDESC"].dropna().unique())
selected_field = st.sidebar.selectbox("Select Field of Study", fields)

# Filter by selected field
df_field = df_school[df_school["CIPDESC"] == selected_field]

# Earnings Metric Filter
metrics = {
    "EARN_MDN_1YR": "1-Year Median Earnings",
    "EARN_MDN_4YR": "4-Year Median Earnings",
    "EARN_MDN_5YR": "5-Year Median Earnings"
}

selected_metric_label = st.sidebar.selectbox(
    "Select Earnings Metric",
    list(metrics.values())
)

# Reverse lookup: label → metric column
selected_metric = [k for k, v in metrics.items() if v == selected_metric_label][0]

# Final filtered dataset
df_filtered = df_field[df_field["EARNINGS_METRIC"] == selected_metric]

# ---------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------
st.title("🎓 Value-Added Earnings by Major")

# Extract the school name from the UNITID-filtered dataset
selected_school_name = df_school["INSTNM"].iloc[0]

st.subheader(f"{selected_field} at {selected_school_name}")

if df_filtered.empty:
    st.warning("No data available for this combination.")
    st.stop()

# ---------------------------------------------------------
# Summary Metrics
# ---------------------------------------------------------
actual = df_filtered["earnings_actual"].iloc[0]
pred = df_filtered["earnings_pred"].iloc[0]
va = df_filtered["earnings_va"].iloc[0]
z = df_filtered["earnings_va_z"].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Actual Earnings", f"${actual:,.0f}")
col2.metric("Predicted Earnings", f"${pred:,.0f}")
col3.metric("Value-Added", f"${va:,.0f}")
col4.metric("VA Z-Score", f"{z:.2f}")

# ---------------------------------------------------------
# Bar Chart: Actual vs Predicted
# ---------------------------------------------------------
chart_df = pd.DataFrame({
    "Type": ["Actual Earnings", "Predicted Earnings"],
    "Value": [actual, pred]
})

bar = (
    alt.Chart(chart_df)
    .mark_bar()
    .encode(
        x=alt.X("Type", sort=None),
        y="Value",
        color="Type"
    )
    .properties(height=350)
)

st.altair_chart(bar, use_container_width=True)

# ---------------------------------------------------------
# Interpretation
# ---------------------------------------------------------
st.markdown("### 📘 Interpretation")

if va > 0:
    st.success(
        f"{selected_school_name} graduates in **{selected_field}** earn "
        f"**${va:,.0f} more** than expected based on institutional characteristics."
    )
else:
    st.error(
        f"{selected_school_name} graduates in **{selected_field}** earn "
        f"**${abs(va):,.0f} less** than expected based on institutional characteristics."
    )

st.info(
    f"**Z-Score {z:.2f}** indicates how far above or below the national average "
    f"this program performs after adjusting for predictors like admissions rate, "
    f"tuition, and student body size."
)
