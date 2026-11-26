"""
baseline.py - Phase 3: prove the cleaning works, then score two 'no-learning' guesses.
These scores are the starting line every real model must beat.
"""
from common import (load_data, learn_fill_values, clean_inputs, make_features,
                    evaluate_model)             # borrow tools from our shared toolbox

train, val = load_data()

# ---- Part 1: did the fixes work? ---------------------------------------------
print("CLEANING CHECK")
fills = learn_fill_values(train)                # typical values learned from the labelled history
for label, df in [("train", train), ("validation", val)]:
    clean = clean_inputs(df, fills)             # apply the fixes
    X = make_features(clean)                    # build the model inputs
    print(f"{label}: negative weights {(df['weight'] < 0).sum()} -> {(clean['weight'] < 0).sum()} | "
          f"blank weights {df['weight'].isna().sum()} -> {clean['weight'].isna().sum()} | "
          f"blank market_index {df['market_index'].isna().sum()} -> {clean['market_index'].isna().sum()} | "
          f"feature table {X.shape}")
    # Safety alarm: stop immediately if any blank is left in the model inputs.
    assert X.isna().sum().sum() == 0, "blanks remain in the features!"


# ---- Part 2: two guesses that involve NO learning ----------------------------
def baseline_quote(fit_df, test_df):
    """Guess = the market's quote per mile x distance."""
    return (test_df["quote_signal"] * test_df["distance"]).values


def baseline_median(fit_df, test_df):
    """Guess = the typical price per mile in the past x distance (ignores everything else)."""
    typical = (fit_df["posted_rate"] / fit_df["distance"]).median()
    return typical * test_df["distance"].values


print("\nROLLING PRACTICE TESTS (each predicts a month it has not seen)")
evaluate_model("baseline_quote_x_distance", baseline_quote, train)
evaluate_model("baseline_median_per_mile", baseline_median, train)

import pandas as pd                             # used only to print the saved table
print("\nSAVED TO results/results.csv (average over 4 tests)")
print(pd.read_csv("results/results.csv").round(2).to_string(index=False))