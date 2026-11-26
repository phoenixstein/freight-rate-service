"""
risk_range.py - Phase 6c: give every load a PRICE RANGE and a SPIKE-RISK score.
Built on recipe D (L1 loss + day-context features) from the stress test.
Evaluated on the four rolling exams Jul-Oct. Reads and prints only.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier   # regressor predicts numbers, classifier predicts yes/no

from common import load_data, make_folds, learn_fill_values, clean_inputs
from recipes import START, with_params, build_features

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

train, _ = load_data()

# Recipe D: L1 loss + day-context features.
FINAL = {**with_params(START, objective="regression_l1"), "day_features": True}
SEED = 42


def predict_range_and_risk(fit_df, test_df):
    """Learn from fit_df; return low / expected / high prices and a spike-risk score for test_df."""
    fills = learn_fill_values(fit_df)                                    # typical values from learning rows only
    fit_clean, test_clean = clean_inputs(fit_df, fills), clean_inputs(test_df, fills)
    X_fit, X_test = build_features(fit_clean, FINAL), build_features(test_clean, FINAL)
    y = np.log(fit_df["posted_rate"] / fit_df["distance"])               # log of price per mile
    miles = test_clean["distance"].values

    out = {}
    # 'Expected' price = recipe D exactly (the middle guess).
    out["expected"] = np.exp(LGBMRegressor(**FINAL["params"]).fit(X_fit, y).predict(X_test)) * miles
    # Low and high = quantile models: the 10th and 90th percentile of the price.
    settings = {k: v for k, v in FINAL["params"].items() if k != "objective"}   # same settings, new objective
    for name, alpha in [("low", 0.10), ("high", 0.90)]:
        model = LGBMRegressor(**settings, objective="quantile", alpha=alpha).fit(X_fit, y)
        out[name] = np.exp(model.predict(X_test)) * miles
    # Safety: make sure low <= expected <= high (separate models can occasionally cross).
    out["low"] = np.minimum(out["low"], out["expected"])
    out["high"] = np.maximum(out["high"], out["expected"])

    # Spike-risk score: probability that a load is a 'spike' (price > 2x or < 0.6x its quote-implied price).
    fit_ratio = fit_df["posted_rate"] / (fit_df["quote_signal"] * fit_df["distance"])
    is_spike = ((fit_ratio > 2.0) | (fit_ratio < 0.6)).astype(int)
    clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                         random_state=SEED, verbose=-1).fit(X_fit, is_spike)
    out["risk"] = clf.predict_proba(X_test)[:, 1]                        # chance (0 to 1) of being a spike
    return pd.DataFrame(out, index=test_df.index)


# ---- Run the four exams and collect every prediction ----------------------------
pieces = []
for label, fit_df, test_df in make_folds(train):
    result = predict_range_and_risk(fit_df, test_df)
    part = test_df[["pickup", "delivery", "distance", "equipment", "quote_signal", "posted_rate"]].join(result)
    part["fold"] = label
    pieces.append(part)
    print("finished exam month", label)
oof = pd.concat(pieces)                                                  # all four months in one table

# Helper columns.
ratio = oof["posted_rate"] / (oof["quote_signal"] * oof["distance"])
oof["spike"] = (ratio > 2.0) | (ratio < 0.6)                             # was this load really a spike?
oof["abs_miss"] = (oof["expected"] - oof["posted_rate"]).abs()           # dollars off for the expected price
oof["inside"] = (oof["posted_rate"] >= oof["low"]) & (oof["posted_rate"] <= oof["high"])  # real price inside the range?

# ---- 1. Sanity check: does 'expected' reproduce recipe D? -----------------------
print("\n1. SANITY CHECK: MAE of the expected price by month (compare D in the stress test: 91.7, 153.2, 94.6, 127.6)")
print(oof.groupby("fold")["abs_miss"].mean().round(1).to_string())

# ---- 2. Is the range honest? (target about 80%) ---------------------------------
print("\n2. RANGE COVERAGE (share of real prices inside the low-high range; target about 80%)")
print(f"overall: {oof['inside'].mean() * 100:.1f}%")
print(f"ordinary loads: {oof.loc[~oof['spike'], 'inside'].mean() * 100:.1f}% | "
      f"spike loads: {oof.loc[oof['spike'], 'inside'].mean() * 100:.1f}%")
print("by month:", (oof.groupby("fold")["inside"].mean() * 100).round(1).to_dict())
width = (oof["high"] - oof["low"]) / oof["expected"] * 100
print(f"typical range width: {width.median():.1f}% of the expected price (median)")

# ---- 3. The review flag: what do we catch if humans check the riskiest X%? ------
print("\n3. REVIEW FLAG (humans check the riskiest share of loads each month)")
# Rank loads within each month by risk: rank_share near 0 = the riskiest loads.
oof["rank_share"] = oof.groupby("fold")["risk"].rank(pct=True, ascending=False)
rows = []
for budget in (0.02, 0.05, 0.10):
    flagged = oof["rank_share"] <= budget                                # the riskiest 'budget' share
    rows.append({
        "loads_reviewed": f"{budget * 100:.0f}%",
        "spikes_caught_%": (flagged & oof["spike"]).sum() / oof["spike"].sum() * 100,   # recall
        "flags_that_were_spikes_%": (flagged & oof["spike"]).sum() / flagged.sum() * 100,  # precision
        "dollar_error_inside_flagged_%": oof.loc[flagged, "abs_miss"].sum() / oof["abs_miss"].sum() * 100,
        "MAE_left_unflagged_$": oof.loc[~flagged, "abs_miss"].mean(),
    })
print(pd.DataFrame(rows).round(1).to_string(index=False))
print(f"(for comparison: all loads MAE ${oof['abs_miss'].mean():.1f}; a RANDOM flag would catch only its own share)")

# ---- 4. What would an analyst see? -----------------------------------------------
print("\n4. THE 6 RISKIEST LOADS, WITH THEIR RANGES")
show = ["pickup", "delivery", "distance", "equipment", "low", "expected", "high", "posted_rate", "risk"]
print(oof.nlargest(6, "risk")[show].round(2).to_string())