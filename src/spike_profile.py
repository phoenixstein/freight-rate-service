"""
spike_profile.py - Phase 6.1: who are the 'spike' loads, and can ANY model see them coming?
Reads only. A 'spike' = a load whose real price is far from what its market quote implies.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier                 # like LGBMRegressor, but predicts yes/no
from sklearn.metrics import roc_auc_score           # the AUC score (0.5 = coin flip, 1.0 = perfect)

from common import load_data, learn_fill_values, clean_inputs, make_features

train, _ = load_data()

# ratio = real price / (quote x distance). Around 1.0 = the quote was about right.
train["ratio"] = train["posted_rate"] / (train["quote_signal"] * train["distance"])
train["month"] = train["date"].dt.to_period("M")
# Label every load: price more than 2x the quote-implied price, less than 0.6x, or ordinary.
train["kind"] = np.where(train["ratio"] > 2.0, "spike_up",
                         np.where(train["ratio"] < 0.6, "spike_down", "ordinary"))
is_spike = train["kind"] != "ordinary"              # True for any spike

print("1. HOW MANY SPIKES?")
print(train["kind"].value_counts().to_string())
print("\nHow big are the 'spike_up' ratios? (is it one fixed multiplier?)")
print(train.loc[train["kind"] == "spike_up", "ratio"]
      .describe(percentiles=[0.1, 0.5, 0.9]).round(2).to_string())

print("\n2. WHAT DO SPIKES LOOK LIKE vs ORDINARY LOADS? (averages)")
print(train.groupby("kind").agg(
    avg_distance=("distance", "mean"), avg_market_index=("market_index", "mean"),
    avg_quote=("quote_signal", "mean")).round(3).to_string())
print("average weight:\n", train["weight"].abs().groupby(train["kind"]).mean().round(0).to_string())
print("\nshare of each kind by trailer type:")
print(pd.crosstab(train["kind"], train["equipment"], normalize="index").round(3).to_string())

print("\n3. DO SPIKES CLUSTER IN TIME OR PLACE?")
print("spike share (%) by month:")
print((is_spike.groupby(train["month"]).mean() * 100).round(2).to_string())
per_day = is_spike.groupby(train["date"]).sum()     # number of spikes on each date
print(f"\nspikes per day: average {per_day.mean():.1f}, busiest day {per_day.max()}")
print("overall spike share (%):", round(is_spike.mean() * 100, 2))
by_city = (is_spike.groupby(train["pickup"]).mean() * 100).sort_values(ascending=False)
print("top 5 pickup cities by spike share (%):\n", by_city.head(5).round(2).to_string())

print("\n4. CAN A MODEL PREDICT SPIKES? (learn on Jan-Aug, test on Sep-Oct)")
fit, test = train[train["date"] < "2025-09-01"], train[train["date"] >= "2025-09-01"]
fills = learn_fill_values(fit)                       # fill-in values from the learning rows only
X_fit = make_features(clean_inputs(fit, fills))
X_test = make_features(clean_inputs(test, fills))
clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                     random_state=42, verbose=-1)
clf.fit(X_fit, is_spike[fit.index].astype(int))      # learn: spike (1) or not (0)
auc = roc_auc_score(is_spike[test.index].astype(int), clf.predict_proba(X_test)[:, 1])
print(f"AUC = {auc:.3f}   (0.5 = coin flip, 1.0 = perfect)")
print("features the model leaned on most:\n",
      pd.Series(clf.feature_importances_, index=X_fit.columns)
      .sort_values(ascending=False).head(5).to_string())