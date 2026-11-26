"""
improve.py - Phase 6.2: the improvement loop.
We try ONE change at a time on top of the current best LightGBM recipe.
A change is KEPT only if it lowers the average MAE on the decision months (Jul, Aug, Sep)
AND helps in at least 2 of those 3 months. October is a locked 'final exam' used ONCE at the end.
Outputs: results/changelog.csv and results/best_config.json
"""
import json                                          # saves the winning recipe as a text file
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from common import (load_data, learn_fill_values, clean_inputs, make_features,
                    make_folds, score)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

train, _ = load_data()

# Fixed thresholds for an 'ordinary' load (middle 98% by price per mile), so we can
# track ordinary loads separately from rare spike loads.
LOW, HIGH = (train["posted_rate"] / train["distance"]).quantile([0.01, 0.99])

# The starting recipe = exactly the Phase 4 winner.
START = {
    "target": "price_per_mile",       # what the model learns
    "day_features": False,            # extra 'what is the day like' columns
    "params": {"objective": "regression", "n_estimators": 300, "learning_rate": 0.05,
               "num_leaves": 31, "random_state": 42, "verbose": -1},
}

all_folds = list(make_folds(train))                  # the 4 rolling exams
decision_folds, final_fold = all_folds[:3], all_folds[3:]   # Jul-Sep to decide, Oct locked


def build_features(df_clean, cfg):
    """Standard features, plus optional day-context columns."""
    X = make_features(df_clean)
    if cfg["day_features"]:
        day_quote = df_clean.groupby("date")["quote_signal"].transform("median")  # the day's typical quote
        X["day_quote_median"] = day_quote
        X["quote_vs_day"] = df_clean["quote_signal"] / day_quote                   # this load's quote vs the day's
    return X


def fit_predict(cfg, fit_df, test_df):
    """Learn from fit_df using recipe cfg, return dollar predictions for test_df."""
    fills = learn_fill_values(fit_df)                          # fill-in values from learning rows only
    fit_clean, test_clean = clean_inputs(fit_df, fills), clean_inputs(test_df, fills)
    X_fit, X_test = build_features(fit_clean, cfg), build_features(test_clean, cfg)
    if cfg["target"] == "quote_correction":
        # Learn how far the real price is from (quote x distance), on a log scale.
        y = np.log(fit_df["posted_rate"] / (fit_df["quote_signal"] * fit_df["distance"]))
        scale = test_clean["quote_signal"] * test_clean["distance"]
    else:
        # Learn the log of price per mile.
        y = np.log(fit_df["posted_rate"] / fit_df["distance"])
        scale = test_clean["distance"]
    model = LGBMRegressor(**cfg["params"]).fit(X_fit, y)     # ** unpacks the settings dictionary
    return np.exp(model.predict(X_test)) * scale.values       # undo the log, scale back to dollars


def evaluate(cfg, folds):
    """Score a recipe on the given exams. One row of results per exam."""
    rows = []
    for _, fit_df, test_df in folds:
        predicted = fit_predict(cfg, fit_df, test_df)
        actual = test_df["posted_rate"].values
        row = score(actual, predicted)
        ordinary = (actual / test_df["distance"].values >= LOW) & (actual / test_df["distance"].values <= HIGH)
        row["ordinary_mae"] = np.abs(predicted[ordinary] - actual[ordinary]).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def with_params(cfg, **new):
    """A copy of a recipe with some model settings changed."""
    return {**cfg, "params": {**cfg["params"], **new}}


# Each change = a name + a function that proposes one or more candidate recipes.
CHANGES = [
    ("1. L1 loss (less swayed by spikes)",
     lambda c: [with_params(c, objective="regression_l1")]),
    ("2. Predict the correction to the quote",
     lambda c: [{**c, "target": "quote_correction"}]),
    ("3. Day-context quote features",
     lambda c: [{**c, "day_features": True}]),
    ("4. Tune settings (best of 4)",
     lambda c: [with_params(c, n_estimators=600, learning_rate=0.03),
                with_params(c, num_leaves=15, min_child_samples=40),
                with_params(c, num_leaves=63, min_child_samples=20),
                with_params(c, n_estimators=600, learning_rate=0.03, num_leaves=15,
                            min_child_samples=40, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.8)]),
]

best_cfg = START
best = evaluate(best_cfg, decision_folds)            # the score to beat
log = [{"change": "0. Start (Phase 4 winner)", "mae_before": np.nan,
        "mae_after": best["mae"].mean(), "mape_after": best["mape"].mean(),
        "ordinary_mae_after": best["ordinary_mae"].mean(), "months_improved": "-", "decision": "start"}]

for name, propose in CHANGES:
    print("trying:", name)
    trials = [(cfg, evaluate(cfg, decision_folds)) for cfg in propose(best_cfg)]
    cfg, res = min(trials, key=lambda t: t[1]["mae"].mean())    # best candidate of this change
    months_better = int((res["mae"].values < best["mae"].values).sum())
    kept = res["mae"].mean() < best["mae"].mean() and months_better >= 2
    log.append({"change": name, "mae_before": best["mae"].mean(), "mae_after": res["mae"].mean(),
                "mape_after": res["mape"].mean(), "ordinary_mae_after": res["ordinary_mae"].mean(),
                "months_improved": f"{months_better}/3", "decision": "KEPT" if kept else "rejected"})
    if kept:                                          # keep it: it becomes the new recipe to beat
        best_cfg, best = cfg, res

changelog = pd.DataFrame(log).round(2)
print("\nCHANGELOG (decision months Jul-Sep, averages)")
print(changelog.to_string(index=False))
changelog.to_csv("results/changelog.csv", index=False)

# ---- The locked final exam: October, used ONCE --------------------------------
print("\nFINAL EXAM (October, never used for decisions)")
before = evaluate(START, final_fold).iloc[0]
after = evaluate(best_cfg, final_fold).iloc[0]
exam = pd.DataFrame({"start_recipe": before, "final_recipe": after}).round(3)
print(exam.to_string())

with open("results/best_config.json", "w") as f:     # save the winning recipe for later phases
    json.dump(best_cfg, f, indent=2)
print("\nFinal recipe saved to results/best_config.json")
print(json.dumps(best_cfg, indent=2))