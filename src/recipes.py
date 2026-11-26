"""
recipes.py - the 'recipe book': turns a settings dictionary (cfg) into predictions.
A recipe has three parts: which target to learn, whether to add day-context features,
and the LightGBM settings. Shared by stress_test.py and, later, the prediction service.
"""
import numpy as np
from lightgbm import LGBMRegressor

from common import learn_fill_values, clean_inputs, make_features   # our shared toolbox

# The Phase 4 winner: our starting recipe.
START = {
    "target": "price_per_mile",       # what the model learns
    "day_features": False,            # extra 'what is the day like' columns
    "params": {"objective": "regression", "n_estimators": 300, "learning_rate": 0.05,
               "num_leaves": 31, "random_state": 42, "verbose": -1},
}


def with_params(cfg, **new):
    """A copy of a recipe with some LightGBM settings changed."""
    return {**cfg, "params": {**cfg["params"], **new}}


def build_features(df_clean, cfg):
    """Standard features, plus optional day-context columns."""
    X = make_features(df_clean)
    if cfg["day_features"]:
        day_quote = df_clean.groupby("date")["quote_signal"].transform("median")  # the day's typical quote
        X["day_quote_median"] = day_quote
        X["quote_vs_day"] = df_clean["quote_signal"] / day_quote                   # this load's quote vs the day's
    return X


def fit_predict(cfg, fit_df, test_df):
    """Learn from fit_df using recipe cfg, then return dollar predictions for test_df."""
    fills = learn_fill_values(fit_df)                          # fill-in values from learning rows only
    fit_clean, test_clean = clean_inputs(fit_df, fills), clean_inputs(test_df, fills)
    X_fit, X_test = build_features(fit_clean, cfg), build_features(test_clean, cfg)
    if cfg["target"] == "quote_correction":
        y = np.log(fit_df["posted_rate"] / (fit_df["quote_signal"] * fit_df["distance"]))
        scale = test_clean["quote_signal"] * test_clean["distance"]
    else:
        y = np.log(fit_df["posted_rate"] / fit_df["distance"])  # log of price per mile
        scale = test_clean["distance"]
    model = LGBMRegressor(**cfg["params"]).fit(X_fit, y)       # ** unpacks the settings dictionary
    return np.exp(model.predict(X_test)) * scale.values         # undo the log, scale back to dollars