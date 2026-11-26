"""
data_quality.py - Phase 2: data-quality audit.
Goal: LOOK for problems in the data. This script only reads and prints.
It never changes the CSV files.
"""
import numpy as np                      # numpy: fast maths on lists of numbers
import pandas as pd                     # pandas: works with tables (rows and columns)
from pathlib import Path               # Path: builds file paths that work on any computer

pd.set_option("display.width", 200)         # allow wider printed tables (less wrapping)
pd.set_option("display.max_columns", None)  # always show every column

DATA = Path("data")  # the folder that holds our CSV files

# Load the two big tables. parse_dates turns the text in "date" into real dates.
train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])  # labelled history (has prices)
val = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])    # future loads (no prices)

# New helper column: price per mile = price / distance (lets us compare short and long trips)
train["rate_per_mile"] = train["posted_rate"] / train["distance"]


def section(title):
    """Print a clear heading so the report is easy to scan."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def r_squared(actual, predicted):
    """R squared: 1 = perfect, 0 = no better than always guessing the average."""
    ss_res = ((actual - predicted) ** 2).sum()        # how far off the predictions are
    ss_tot = ((actual - actual.mean()) ** 2).sum()    # how far off "just guess the average" is
    return 1 - ss_res / ss_tot


def haversine_miles(lat1, lon1, lat2, lon2):
    """Straight-line miles between two latitude/longitude points on Earth."""
    p1, p2 = np.radians(lat1), np.radians(lat2)       # degrees -> radians (the formula needs radians)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))         # 3958.8 = Earth's radius in miles


# ---- 1. Duplicates and impossible values -------------------------------------
section("1. DUPLICATES AND IMPOSSIBLE VALUES")
no_id = [c for c in train.columns if c not in ("load_id", "rate_per_mile")]  # compare rows ignoring the ID
print("Duplicate load IDs:", train["load_id"].duplicated().sum())           # should be 0
print("Duplicate rows (ignoring ID):", train.duplicated(subset=no_id).sum())  # should be 0
print("Price <= 0:", (train["posted_rate"] <= 0).sum())                     # a price can't be zero or negative
print("Distance <= 0 (train / validation):",
      (train["distance"] <= 0).sum(), "/", (val["distance"] <= 0).sum())
print("Pickup same as delivery (train / validation):",
      (train["pickup"] == train["delivery"]).sum(), "/", (val["pickup"] == val["delivery"]).sum())

# ---- 2. Weight ---------------------------------------------------------------
section("2. WEIGHT (a load can't weigh less than zero)")
for name, df in [("train", train), ("validation", val)]:
    w = df["weight"]
    print(f"{name}: blank={w.isna().sum()}  negative={(w < 0).sum()}  "
          f"min={w.min():.0f}  max={w.max():.0f}")
# If the negatives, once made positive, look like normal weights, they are probably sign typos.
print("Negative weights made positive - typical (median):",
      round(train.loc[train["weight"] < 0, "weight"].abs().median()))
print("Normal positive weights - typical (median):",
      round(train.loc[train["weight"] > 0, "weight"].median()))

# ---- 3. Blank cells: random or patterned? -----------------------------------
section("3. BLANK CELLS: RANDOM OR PATTERNED?")
for col in ["weight", "market_index"]:
    by_month = train[col].isna().groupby(train["date"].dt.to_period("M")).mean()  # share blank per month
    print(f"{col}: share blank - train {train[col].isna().mean():.4f}, "
          f"validation {val[col].isna().mean():.4f}; "
          f"by month ranges {by_month.min():.4f} to {by_month.max():.4f}")

# ---- 4. Is market_index a daily number? -------------------------------------
section("4. MARKET_INDEX: ONE NUMBER PER DAY, OR PER LOAD?")
within = train.groupby("date")["market_index"].std().mean()   # how much it varies inside one day
between = train.groupby("date")["market_index"].mean().std()  # how much the daily averages vary
print(f"Variation inside a day: {within:.4f} | variation between days: {between:.4f}")
print("If 'inside' is much smaller, a blank can be filled using other loads from the same day.")

# ---- 5. The price we want to predict ----------------------------------------
section("5. PRICE AND RATE PER MILE (look for outliers)")
q = [0.01, 0.5, 0.99, 0.999]  # which percentiles to show
print(train["posted_rate"].describe(percentiles=q).round(2).to_string())
print(train["rate_per_mile"].describe(percentiles=q).round(3).to_string())
show = ["load_id", "pickup", "delivery", "distance", "equipment", "posted_rate", "rate_per_mile"]
print("\n5 most expensive per mile:\n", train.nlargest(5, "rate_per_mile")[show].to_string())
print("\n5 cheapest per mile:\n", train.nsmallest(5, "rate_per_mile")[show].to_string())

# ---- 6. The "no-learning" baseline ------------------------------------------
section("6. BASELINE: quote_signal x distance (no learning at all)")
guess = train["quote_signal"] * train["distance"]   # quote per mile x miles = a dollar guess
actual = train["posted_rate"]
miss = guess - actual                               # how far each guess is from the truth
print(f"MAE  (average miss, $):          {miss.abs().mean():.2f}")
print(f"MAPE (average miss, % of price): {(miss.abs() / actual).mean() * 100:.2f}%")
print(f"RMSE (like MAE, big misses hurt more): {np.sqrt((miss ** 2).mean()):.2f}")
print(f"R squared:                       {r_squared(actual, guess):.4f}")

# ---- 7. Do distances agree with coordinates? ---------------------------------
section("7. DISTANCE vs STRAIGHT-LINE DISTANCE FROM COORDINATES")
for name, df in [("train", train), ("validation", val)]:
    straight = haversine_miles(df["pickup_lat"], df["pickup_lon"],
                               df["delivery_lat"], df["delivery_lon"])
    ratio = df["distance"] / straight   # road miles are normally ~1.2x the straight line
    print(f"{name}: ratio median {ratio.median():.3f}, 99th percentile {ratio.quantile(0.99):.3f}, "
          f"max {ratio.max():.1f}, rows above 2.0: {(ratio > 2.0).sum()}")

# ---- 8. Cities and the December lane ----------------------------------------
section("8. CITIES SEEN ONLY IN THE FUTURE DATA")
train_cities = set(train["pickup"]) | set(train["delivery"])   # every city in training
val_cities = set(val["pickup"]) | set(val["delivery"])         # every city in validation
print("Cities only in validation:", sorted(val_cities - train_cities))
lane = train[(train["pickup"] == "Lexington") & (train["delivery"] == "Fort Wayne")]
print("Past Lexington -> Fort Wayne loads in training:", len(lane))

# ---- 9. Has the market changed over time? ------------------------------------
section("9. DRIFT: AVERAGES BY MONTH")
print(train.groupby(train["date"].dt.to_period("M")).agg(
    loads=("load_id", "count"), market_index=("market_index", "mean"),
    quote_signal=("quote_signal", "mean"), rate_per_mile=("rate_per_mile", "mean")).round(3).to_string())
print("\nValidation market_index range:", round(val["market_index"].min(), 3),
      "to", round(val["market_index"].max(), 3),
      "| training range:", round(train["market_index"].min(), 3),
      "to", round(train["market_index"].max(), 3))