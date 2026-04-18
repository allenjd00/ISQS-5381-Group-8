# -*- coding: utf-8 -*-
"""
College price trend graphs for one school compared to yearly medians
"""

import pandas as pd
import matplotlib.pyplot as plt


file_path = r"D:\Senior Project\Cost Data\panel_costs.csv"
df = pd.read_csv(file_path)

#school select
school_id = 100654

school_df = df[df["unitid"] == school_id].copy()
school_df = school_df.sort_values("Year")
school_df = school_df.reset_index(drop=True)

#all school median
median_df = df.groupby("Year")[["ISPrice", "OOSPrice"]].median().reset_index()

median_df = median_df.rename(columns={
    "ISPrice": "Median_ISPrice",
    "OOSPrice": "Median_OOSPrice"
})

#add yr
comb_df = pd.merge(school_df, median_df, on="Year", how="left")

#% change
comb_df["IS_YOY"] = comb_df["ISPrice"].pct_change() * 100
comb_df["Median_IS_YOY"] = comb_df["Median_ISPrice"].pct_change() * 100

comb_df["OOS_YOY"] = comb_df["OOSPrice"].pct_change() * 100
comb_df["Median_OOS_YOY"] = comb_df["Median_OOSPrice"].pct_change() * 100

#$ difference from median
comb_df["IS_Dollar_Diff"] = comb_df["ISPrice"] - comb_df["Median_ISPrice"]
comb_df["OOS_Dollar_Diff"] = comb_df["OOSPrice"] - comb_df["Median_OOSPrice"]


# Price vs median, graph 1&2
plt.figure(figsize=(10, 6))
plt.plot(comb_df["Year"], comb_df["ISPrice"], marker="o", label="School In-State Price")
plt.plot(comb_df["Year"], comb_df["Median_ISPrice"], marker="o", label="In-State Median")
plt.title("In-State Price vs Median \nunitid " + str(school_id))
plt.xlabel("Year")
plt.ylabel("$, Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(comb_df["Year"], comb_df["OOSPrice"], marker="o", label="School Out-of-State Price")
plt.plot(comb_df["Year"], comb_df["Median_OOSPrice"], marker="o", label="Out-of-State Median")
plt.title("Out-of-State Price vs Median \nunitid " + str(school_id))
plt.xlabel("Year")
plt.ylabel("$, Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# $ Vs  median, graph 5&6
plt.figure(figsize=(10, 6))
plt.plot(comb_df["Year"], comb_df["IS_Dollar_Diff"], marker="o", label="ISPrice - Median")
plt.axhline(0, linestyle="--", color="orange", label="Median")
plt.title("In-State Price, $ from Median\nunitid " + str(school_id))
plt.xlabel("Year")
plt.ylabel("Delta $")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(comb_df["Year"], comb_df["OOS_Dollar_Diff"], marker="o", label="OOSPrice - Median")
plt.axhline(0, linestyle="--", color="orange", label="Median")
plt.title("Out-of-State, $ from Median\nunitid " + str(school_id))
plt.xlabel("Year")
plt.ylabel(" Delta $")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# YoY percent change(rate of change) v median, graph 3 and 4
plt.figure(figsize=(10, 6))
plt.plot(comb_df["Year"], comb_df["IS_YOY"], marker="o", label="School In-State YoY %")
plt.plot(comb_df["Year"], comb_df["Median_IS_YOY"], marker="o", label="Median In-State YoY %")
plt.axhline(0, linestyle="--")
plt.title("In-State Year-over-Year % Delta vs Median\nunitid " + str(school_id))
plt.xlabel("Year")
plt.ylabel("% Change")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(comb_df["Year"], comb_df["OOS_YOY"], marker="o", label="School Out-of-State YoY %")
plt.plot(comb_df["Year"], comb_df["Median_OOS_YOY"], marker="o", label="Median Out-of-State YoY %")
plt.axhline(0, linestyle="--")
plt.title("Out-of-State Year-over-Year % Delta vs Median\nunitid " + str(school_id))
plt.xlabel("Year")
plt.ylabel("% Change ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

