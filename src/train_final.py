"""
train_final.py - Phase 7a: final model, final predictions, and an honest price range.
Final recipe = D (L1 loss + day-context features), chosen over the strict rule's pick (A)
because it was ~10% better on average over seven months. This choice is disclosed in the report.
Outputs: validation_predictions.csv, data/december_chart_inputs.csv (filled),
         results/best_config.json, results/range_calibration.json
"""
import json                                   # saves small settings files
import numpy as np
import pandas as pd

from common import load_data, make_folds, DATA
from recipes import START, with_params, fit_predict   # fit_predict(recipe, learn_rows, rows_to_predict)

train, val = load_data()
template = pd.read_csv(DATA / "validation_predictions_template.csv")   # defines the required row order
dec_raw = pd.read_csv(DATA / "december_chart_inputs.csv")              # the 31 chart rows

# The final recipe (D): L1 loss + day-context features.
FINAL = {**with_params(START, objective="regression_l1"), "day_features": True}
with open("results/best_config.json", "w") as f:       # save it so later phases use the same recipe
    json.dump(FINAL, f, indent=2)

# ---- 1. Predict the 12,000 validation loads --------------------------------------
val_pred = fit_predict(FINAL, train, val)               # learn from ALL 48,000 loads, predict the future ones
predictions = pd.DataFrame({"load_id": val["load_id"].values,
                            "predicted_rate": np.clip(val_pred, 1.0, None).round(2)})  # the scorer needs rates > 0
predictions = template[["load_id"]].merge(predictions, on="load_id", how="left")      # follow the template's order
# Safety alarms: stop if anything is off.
assert len(predictions) == 12000, "expected exactly 12,000 rows"
assert predictions["predicted_rate"].notna().all(), "some loads have no prediction"
assert (predictions["predicted_rate"] > 0).all(), "all predictions must be positive"
predictions.to_csv("validation_predictions.csv", index=False)
print("1. SAVED validation_predictions.csv:", predictions.shape,
      "| min", predictions["predicted_rate"].min(),
      "| median", round(predictions["predicted_rate"].median(), 2),
      "| max", predictions["predicted_rate"].max())

# ---- 2. How different are the strict-rule pick (A) and our choice (D)? -----------
a_pred = fit_predict(START, train, val)                 # the same job with recipe A
diff = np.abs(val_pred - a_pred) / a_pred * 100         # % difference per load
print(f"\n2. RECIPE A vs D on the 12,000 loads: average difference {diff.mean():.1f}% | "
      f"loads differing by more than 5%: {(diff > 5).mean() * 100:.1f}%")

# ---- 3. December chart predictions -----------------------------------------------
dec = dec_raw.copy()
dec["date"] = pd.to_datetime(dec_raw["date"])
# The chart file has no market columns. Use each day's typical value from validation loads (inputs, not answers).
daily = val.groupby("date")[["market_index", "quote_signal"]].median()
dec["market_index"] = dec["date"].map(daily["market_index"]).fillna(val["market_index"].median())
dec["quote_signal"] = dec["date"].map(daily["quote_signal"]).fillna(val["quote_signal"].median())
# It also has no coordinates, so look each city up from the training data.
cities = pd.concat([
    train[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
    train[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
]).drop_duplicates("city").set_index("city")
dec["pickup_lat"], dec["pickup_lon"] = dec["pickup"].map(cities["lat"]), dec["pickup"].map(cities["lon"])
dec["delivery_lat"], dec["delivery_lon"] = dec["delivery"].map(cities["lat"]), dec["delivery"].map(cities["lon"])
dec_raw["predicted_rate"] = np.clip(fit_predict(FINAL, train, dec), 1.0, None).round(2)  # fill the blank column
dec_raw.to_csv(DATA / "december_chart_inputs.csv", index=False)       # same 7 columns, same order
print("\n3. DECEMBER PREDICTIONS (Lexington -> Fort Wayne)")
print(dec_raw[["date", "predicted_rate"]].to_string(index=False))

# ---- 4. Honest price range: use how wrong the model really was ------------------
ratios = []
for label, fit_df, test_df in make_folds(train):               # the four rolling exams, Jul-Oct
    predicted = fit_predict(FINAL, fit_df, test_df)
    # ratio = real price / our prediction. 1.0 = perfect, 1.1 = real price was 10% higher.
    ratios.append(pd.DataFrame({"fold": label, "ratio": test_df["posted_rate"].values / predicted}))
r = pd.concat(ratios)

print("\n4. CALIBRATED RANGE CHECK (range set from 3 months, tested on the 4th; target 80%)")
for label in r["fold"].unique():
    others = r.loc[r["fold"] != label, "ratio"]                 # the other three months
    low, high = others.quantile(0.10), others.quantile(0.90)    # the 10th and 90th percentile of past ratios
    this = r.loc[r["fold"] == label, "ratio"]
    print(f"  {label}: range x[{low:.3f}, {high:.3f}] covers {((this >= low) & (this <= high)).mean() * 100:.1f}%")

# Final multipliers use all four months and are saved for the prediction service later.
low, high = r["ratio"].quantile(0.10), r["ratio"].quantile(0.90)
with open("results/range_calibration.json", "w") as f:
    json.dump({"low_multiplier": round(float(low), 4), "high_multiplier": round(float(high), 4),
               "meaning": "range = expected price x [low, high]; aimed at ~80% coverage"}, f, indent=2)
print(f"\nFINAL RANGE: expected price x [{low:.3f}, {high:.3f}]  "
      f"(for a $2,000 prediction: ${2000 * low:,.0f} to ${2000 * high:,.0f})")