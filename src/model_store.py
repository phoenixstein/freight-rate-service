# model_store.py - trains the FINAL recipe once, saves it to a file, and loads it back for predictions.
# Run directly to train, save, and verify against validation_predictions.csv.

import joblib                          # saves and loads Python objects (like a trained model)
import numpy as np                     # maths on arrays
import pandas as pd                    # tables
from pathlib import Path               # safe file paths
from lightgbm import LGBMRegressor     # the model type used by recipe D

from common import load_data, learn_fill_values, clean_inputs     # shared cleaning tools
from recipes import START, with_params, build_features            # the recipe machinery

# The final recipe D: L1 loss plus day-context features (same as train_final.py).
FINAL = {**with_params(START, objective="regression_l1"), "day_features": True}

# The calibrated price range from results/range_calibration.json (see the train_final log).
RANGE_LOW, RANGE_HIGH = 0.986, 1.070

MODEL_FILE = Path("models/rate_model.joblib")    # where the frozen model lives (ignored by Git)


def train_and_save():
    """Train recipe D on ALL labelled loads, then save everything needed to predict later."""
    train, _ = load_data()                                       # labelled history
    fills = learn_fill_values(train)                             # typical values for blanks
    clean = clean_inputs(train, fills)                           # apply the Phase 2 fixes
    X = build_features(clean, FINAL)                             # the model's input columns
    y = np.log(train["posted_rate"] / train["distance"])         # target: log of price per mile
    model = LGBMRegressor(**FINAL["params"]).fit(X, y)           # train the model
    bundle = {
        "model": model,                                          # the trained model
        "fills": fills,                                          # how to fill blanks
        "feature_columns": list(X.columns),                      # exact column order it expects
        "range": (RANGE_LOW, RANGE_HIGH),                        # the price range multipliers
    }
    MODEL_FILE.parent.mkdir(exist_ok=True)                       # make the models folder if needed
    joblib.dump(bundle, MODEL_FILE)                              # write the bundle to disk
    return bundle


def load_bundle():
    """Load the saved bundle from disk."""
    return joblib.load(MODEL_FILE)


def predict_loads(bundle, df):
    """Dollar price predictions for each row of df (same cleaning and features as training)."""
    clean = clean_inputs(df, bundle["fills"])                    # same fixes as training
    X = build_features(clean, FINAL)[bundle["feature_columns"]]  # same columns, same order
    per_mile = np.exp(bundle["model"].predict(X))                # undo the log: dollars per mile
    return np.clip(per_mile * clean["distance"].values, 1.0, None)   # times miles; never below $1

def predict_with_day_median(bundle, df, day_quote_median):
    """Like predict_loads, but the day's typical quote is GIVEN (looked up), not computed from df.
    Needed for single requests: one row cannot work out a daily median by itself."""
    clean = clean_inputs(df, bundle["fills"])                    # same fixes as training
    X = build_features(clean, {"day_features": False})           # the standard columns only
    X["day_quote_median"] = day_quote_median                     # the looked-up typical quote that day
    X["quote_vs_day"] = clean["quote_signal"] / day_quote_median # this load's quote vs the day's
    X = X[bundle["feature_columns"]]                             # same column order as training
    per_mile = np.exp(bundle["model"].predict(X))                # undo the log: dollars per mile
    return np.clip(per_mile * clean["distance"].values, 1.0, None)   # times miles; never below $1

if __name__ == "__main__":
    train_and_save()                                             # 1. train and save
    print(f"Saved {MODEL_FILE} ({MODEL_FILE.stat().st_size / 1e6:.1f} MB)")
    bundle = load_bundle()                                       # 2. load it back from disk
    _, val = load_data()
    mine = pd.DataFrame({"load_id": val["load_id"], "mine": predict_loads(bundle, val)})
    official = pd.read_csv("validation_predictions.csv")         # the file we submitted
    merged = official.merge(mine, on="load_id")                  # line them up by load_id
    biggest = (merged["predicted_rate"] - merged["mine"]).abs().max()   # biggest disagreement in $
    print("rows compared:", len(merged), "| biggest difference: $", round(biggest, 4))
    # The submitted file is rounded to cents, so up to half a cent of difference is expected.
    print("MATCH" if len(merged) == 12000 and biggest < 0.01 else "MISMATCH")
    low, high = bundle["range"]                                  # 3. show an example with its range
    first = merged.iloc[0]
    print(f"example: {first['load_id']} predicted ${first['mine']:.2f}, "
          f"range ${first['mine'] * low:.2f} to ${first['mine'] * high:.2f}")