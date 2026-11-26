"""
stress_test.py - Phase 6b: do the improvements hold up month after month?
Phase 6.2 picked changes using Jul-Sep, and the locked October exam did NOT confirm them.
Here five recipes face SEVEN rolling exams (Apr-Oct), so no single month can decide.
NOTE: October has now been looked at, so no untouched exam is left in this data. We say so in the report.
Outputs: results/stress_test.csv and (overwrites) results/best_config.json
"""
import json
import numpy as np
import pandas as pd

from common import load_data, score
from recipes import START, with_params, fit_predict

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

train, val = load_data()
# Fixed thresholds for an 'ordinary' load (middle 98% by price per mile).
LOW, HIGH = (train["posted_rate"] / train["distance"]).quantile([0.01, 0.99])

# Edges of the 7 exams: each month Apr..Oct is predicted using only EARLIER months.
BOUNDS = pd.to_datetime(["2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01",
                         "2025-08-01", "2025-09-01", "2025-10-01", "2025-11-01"])

# The five recipes. Each differs from the one before it by one idea.
L1 = with_params(START, objective="regression_l1")
RECIPES = {
    "A start (L2)": START,
    "B + L1 loss": L1,
    "C + day features": {**START, "day_features": True},
    "D L1 + day features": {**L1, "day_features": True},
    "E D + tuned": {**with_params(L1, n_estimators=600, learning_rate=0.03), "day_features": True},
}

rows = []
for i in range(7):                                              # one pass per exam month
    month = BOUNDS[i].strftime("%m-%b")                         # e.g. '04-Apr' (sorts in time order)
    fit_df = train[train["date"] < BOUNDS[i]]                   # everything BEFORE the exam month
    test_df = train[(train["date"] >= BOUNDS[i]) & (train["date"] < BOUNDS[i + 1])]
    actual = test_df["posted_rate"].values
    per_mile = actual / test_df["distance"].values
    ordinary = (per_mile >= LOW) & (per_mile <= HIGH)           # True for ordinary loads
    for name, cfg in RECIPES.items():
        predicted = fit_predict(cfg, fit_df, test_df)
        s = score(actual, predicted)
        rows.append({"recipe": name, "month": month, "mae": s["mae"], "mape": s["mape"],
                     "ordinary_mae": np.abs(predicted[ordinary] - actual[ordinary]).mean()})
    print("finished exam month", month)

long = pd.DataFrame(rows)
long.to_csv("results/stress_test.csv", index=False)

# ---- Tables: recipes down the side, months across the top ----------------------
mae = long.pivot(index="recipe", columns="month", values="mae")
ord_mae = long.pivot(index="recipe", columns="month", values="ordinary_mae")
mae["average"], ord_mae["average"] = mae.mean(axis=1), ord_mae.mean(axis=1)
print("\nMAE ($) BY EXAM MONTH")
print(mae.round(1).to_string())
print("\nMAE ($) ON ORDINARY LOADS ONLY")
print(ord_mae.round(1).to_string())

# ---- Summary and the pre-agreed decision rule ----------------------------------
base = mae.loc["A start (L2)"].drop("average")                  # the starting recipe's monthly MAE
summary = []
for name in mae.index:
    monthly = mae.loc[name].drop("average")
    summary.append({"recipe": name, "avg_mae": monthly.mean(),
                    "avg_mape": long.loc[long["recipe"] == name, "mape"].mean(),
                    "months_better_than_A": int((monthly < base).sum())})
summary = pd.DataFrame(summary)
print("\nSUMMARY (7 exam months)")
print(summary.round(2).to_string(index=False))

# Eligible = beats A in at least 5 of 7 months. Winner = lowest average MAE among the eligible.
eligible = summary[(summary["recipe"] != "A start (L2)") & (summary["months_better_than_A"] >= 5)]
chosen = (eligible.sort_values("avg_mae").iloc[0]["recipe"] if len(eligible)
          else "A start (L2)")                                  # nobody qualified -> keep the start
print("\nCHOSEN BY THE RULE:", chosen)
with open("results/best_config.json", "w") as f:                # save the chosen recipe for Phase 7
    json.dump(RECIPES[chosen], f, indent=2)

# ---- Bonus: do low-quote loads track spikes, and what do Nov-Dec look like? ----
ratio = train["posted_rate"] / (train["quote_signal"] * train["distance"])
by_month = train["date"].dt.to_period("M")
table = pd.DataFrame({
    "low_quote_%": (train["quote_signal"] < 1.7).groupby(by_month).mean() * 100,   # share of loads with quote < 1.7
    "spike_%": ((ratio > 2) | (ratio < 0.6)).groupby(by_month).mean() * 100,       # share of spike loads
})
print("\nTRAIN: low-quote share vs spike share by month")
print(table.round(1).to_string())
print("\nVALIDATION: low-quote share by month (no prices to check spikes)")
print(((val["quote_signal"] < 1.7).groupby(val["date"].dt.to_period("M")).mean() * 100).round(1).to_string())