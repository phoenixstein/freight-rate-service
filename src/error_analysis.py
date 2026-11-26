"""
error_analysis.py - Phase 5: WHERE does the winning model (LightGBM) make its mistakes?
It re-runs the 4 practice exams, collects every prediction, and then slices the errors
by price type, distance, trailer type and month. It only reads and prints.
"""
import pandas as pd
from lightgbm import LGBMRegressor

from common import load_data, make_folds, score      # our shared toolbox
from runner import make_runner                       # the learn-then-predict recipe

train, _ = load_data()                               # labelled history only

# The winning model, with the same settings used in the race.
run = make_runner(lambda: LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                        num_leaves=31, random_state=42, verbose=-1))

# Collect "out-of-fold" predictions: each row is predicted by a model that never saw its month.
pieces = []
for label, fit_df, test_df in make_folds(train):
    part = test_df.copy()
    part["predicted"] = run(fit_df, test_df)         # predict the test month
    pieces.append(part)
oof = pd.concat(pieces)                              # glue the 4 months into one table

# Helper columns for the analysis.
oof["miss"] = (oof["predicted"] - oof["posted_rate"]).abs()          # dollars off, ignoring direction
oof["pct_miss"] = oof["miss"] / oof["posted_rate"] * 100             # miss as % of the true price
oof["rate_per_mile"] = oof["posted_rate"] / oof["distance"]          # price per mile
oof["month"] = oof["date"].dt.to_period("M")                         # e.g. 2025-07

# ---- 1. Where does the big squared error come from? ---------------------------
print("1. WHERE DOES RMSE COME FROM?")
squared = (oof["miss"] ** 2).sort_values(ascending=False)            # biggest squared misses first
for share in (0.001, 0.01, 0.05):
    n = int(len(squared) * share)                                    # how many loads in the worst X%
    print(f"  the worst {share * 100:g}% of loads ({n} loads) cause "
          f"{squared.iloc[:n].sum() / squared.sum() * 100:.1f}% of all squared error")

# ---- 2. Ordinary loads vs spike/odd loads -------------------------------------
print("\n2. ORDINARY LOADS vs SPIKE/ODD LOADS (split by price per mile)")
low, high = oof["rate_per_mile"].quantile([0.01, 0.99])              # the 1st and 99th percentile
ordinary = oof["rate_per_mile"].between(low, high)                   # True for the middle 98%
for label, rows in [("ordinary (middle 98%)", oof[ordinary]),
                    ("spike/odd (outer 2%)", oof[~ordinary])]:
    s = score(rows["posted_rate"], rows["predicted"])
    print(f"  {label:<22} {len(rows):>5} loads | MAE ${s['mae']:8.2f} | "
          f"MAPE {s['mape']:6.2f}% | RMSE ${s['rmse']:8.2f} | R2 {s['r2']:.3f}")

# ---- 3. Errors by group ---------------------------------------------------------
oof["distance_band"] = pd.cut(oof["distance"], [0, 250, 500, 1000, 2000, 4000],
                              labels=["under 250", "250-500", "500-1000", "1000-2000", "2000+"])
for title, col in [("3a. BY DISTANCE BAND", "distance_band"),
                   ("3b. BY TRAILER TYPE", "equipment"),
                   ("3c. BY MONTH", "month")]:
    print(f"\n{title}")
    print(oof.groupby(col, observed=True).agg(
        loads=("miss", "size"), mae=("miss", "mean"), mape=("pct_miss", "mean")
    ).round(2).to_string())

# ---- 4. The 8 biggest single misses --------------------------------------------
print("\n4. THE 8 BIGGEST MISSES")
cols = ["pickup", "delivery", "distance", "equipment", "date", "posted_rate", "predicted", "rate_per_mile"]
print(oof.nlargest(8, "miss")[cols].round(2).to_string())