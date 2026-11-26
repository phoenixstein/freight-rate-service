"""
business_simulation.py - Phase 8a: what is the model worth to a business?
We play a freight broker who must quote a price BEFORE knowing what the truck costs.
Same loads, same rules, four different 'price guessers'. All guesses come from the rolling
exams (each month predicted using only EARLIER months), so no guesser sees the answer.
ASSUMPTIONS (margin, tolerance) are illustrative; we test many values to see if the conclusion holds.
"""
import numpy as np
import pandas as pd

from common import load_data, make_folds
from recipes import START, with_params, fit_predict

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

train, _ = load_data()
FINAL = {**with_params(START, objective="regression_l1"), "day_features": True}   # recipe D
# 'Ordinary' loads = middle 98% by price per mile. The other 2% are rare spikes.
LOW, HIGH = (train["posted_rate"] / train["distance"]).quantile([0.01, 0.99])

GUESSERS = ["Gut feel (median $/mile)", "Posted quote x distance", "Basic model (A)", "Final model (D)"]
CEILING = "Perfect knowledge (ceiling)"

# ---- 1. Collect every guesser's price guess for the Jul-Oct loads ---------------
pieces = []
for label, fit_df, test_df in make_folds(train):
    part = test_df[["distance", "posted_rate"]].copy()
    part["month"] = label
    typical = (fit_df["posted_rate"] / fit_df["distance"]).median()      # the past typical $ per mile
    part[GUESSERS[0]] = typical * test_df["distance"].values             # gut feel: same $/mile for everyone
    part[GUESSERS[1]] = (test_df["quote_signal"] * test_df["distance"]).values   # the posted market quote
    part[GUESSERS[2]] = fit_predict(START, fit_df, test_df)              # basic model
    part[GUESSERS[3]] = fit_predict(FINAL, fit_df, test_df)              # final model
    part[CEILING] = test_df["posted_rate"].values                        # cheating: knows the true cost
    pieces.append(part)
    print("finished month", label)
sim = pd.concat(pieces)

cost = sim["posted_rate"].values                       # what the truck REALLY costs the broker
per_mile = cost / sim["distance"].values
ordinary = (per_mile >= LOW) & (per_mile <= HIGH)      # True for ordinary loads
months = sim["month"].values


def simulate(guess, margin, tolerance):
    """Play the broker. Returns the profit on every load and whether the customer accepted."""
    quote = guess * (1 + margin)                       # the broker quotes the guess plus a margin
    accepted = quote <= cost * (1 + tolerance)         # the customer accepts if it is not too far above the true cost
    profit = np.where(accepted, quote - cost, 0.0)     # accepted: quote - cost (can be negative!); rejected: 0
    return profit, accepted


# ---- 2. Headline comparison under one set of assumptions -------------------------
MARGIN, TOLERANCE = 0.05, 0.10
rows = []
for name in GUESSERS + [CEILING]:
    profit, accepted = simulate(sim[name].values, MARGIN, TOLERANCE)
    rows.append({
        "guesser": name,
        "loads_won_%": accepted[ordinary].mean() * 100,
        "wins_below_cost_%": ((profit < 0) & ordinary).sum() / ordinary.sum() * 100,   # won the load but lost money
        "profit_per_load_$": profit[ordinary].mean(),
        "total_profit_$": profit[ordinary].sum(),
        "spike_loads_total_$": profit[~ordinary].sum(),                                 # the rare big-surprise loads
    })
table = pd.DataFrame(rows)
print(f"\n1. HEADLINE (margin {MARGIN:.0%}, tolerance {TOLERANCE:.0%}) - ordinary loads, Jul-Oct, "
      f"{ordinary.sum():,} loads")
print(table.round(1).to_string(index=False))

by = table.set_index("guesser")["profit_per_load_$"]
final, gut, basic, ceil = by["Final model (D)"], by[GUESSERS[0]], by["Basic model (A)"], by[CEILING]
print(f"\nFinal model vs gut feel:   ${(final - gut) * 1000:,.0f} more profit per 1,000 loads")
print(f"Final model vs basic model: ${(final - basic) * 1000:,.0f} more profit per 1,000 loads")
print(f"Final model captures {final / ceil * 100:.0f}% of the perfect-knowledge ceiling "
      f"(gut feel: {gut / ceil * 100:.0f}%)")

# ---- 3. Does the conclusion depend on my assumptions? ---------------------------
grid = {}
for margin in (0.03, 0.05, 0.08):
    for tol in (0.05, 0.10, 0.15):
        key = f"m{margin:.0%}/t{tol:.0%}"
        grid[key] = {n: simulate(sim[n].values, margin, tol)[0][ordinary].mean() for n in GUESSERS}
grid = pd.DataFrame(grid)
print("\n2. PROFIT PER ORDINARY LOAD ($) UNDER 9 DIFFERENT ASSUMPTION SETS (m=margin, t=tolerance)")
print(grid.round(1).to_string())
best_count = int((grid.loc["Final model (D)"] == grid.max()).sum())   # columns where D is the top real guesser
print(f"Final model is the best of the four guessers in {best_count} of {grid.shape[1]} assumption sets")

# ---- 4. Is the result steady month by month? -------------------------------------
print(f"\n3. PROFIT PER ORDINARY LOAD ($) BY MONTH (margin {MARGIN:.0%}, tolerance {TOLERANCE:.0%})")
monthly = {}
for name in GUESSERS:
    profit, _ = simulate(sim[name].values, MARGIN, TOLERANCE)
    monthly[name] = {m: profit[ordinary & (months == m)].mean() for m in sorted(set(months))}
print(pd.DataFrame(monthly).round(1).to_string())