"""
common.py - shared tools used by every later script ("one kitchen, many recipes").
Contains: loading data, cleaning, building features, scoring, and the 4 rolling
time-based practice tests (folds) used to compare models fairly.
"""
import time                          # lets us measure how many seconds something takes
import numpy as np                   # fast maths on lists of numbers
import pandas as pd                  # tables of data
from pathlib import Path             # builds file paths that work on any computer

DATA = Path("data")                  # folder with the CSV files
RESULTS = Path("results")            # folder where we save the comparison table
EQUIPMENT_TYPES = ["Dry Van", "Flatbed", "Reefer"]   # the 3 trailer types in the data

# The edges of our 4 practice tests. Test k predicts the month between two edges
# using ONLY rows from before the first edge (so a model never sees its own test month).
FOLD_BOUNDS = pd.to_datetime(["2025-07-01", "2025-08-01", "2025-09-01",
                              "2025-10-01", "2025-11-01"])


def load_data():
    """Read the labelled history (train) and the future loads (validation)."""
    train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])  # has posted_rate (the answer)
    val = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])    # no answers: we predict these
    return train, val


def haversine_miles(lat1, lon1, lat2, lon2):
    """Straight-line miles between two latitude/longitude points on Earth."""
    p1, p2 = np.radians(lat1), np.radians(lat2)        # degrees -> radians
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))          # 3958.8 = Earth's radius in miles


def learn_fill_values(fit_df):
    """Work out the 'typical' values used to fill blanks.
    IMPORTANT: computed from the rows we learn from only, so no test information leaks in."""
    weight_medians = fit_df["weight"].abs().groupby(fit_df["equipment"]).median().to_dict()  # typical weight per trailer type
    market_fallback = fit_df["market_index"].median()   # last-resort market_index if a whole day is blank
    return {"weight_medians": weight_medians, "market_fallback": market_fallback}


def clean_inputs(df, fills):
    """Apply the Phase 2 fixes. Works on a copy, so the original table is never changed."""
    df = df.copy()
    df["weight"] = df["weight"].abs()                                    # negative weights are sign typos -> positive
    df["weight_missing"] = df["weight"].isna().astype(int)               # remember which weights were blank (1 = blank)
    df["weight"] = df["weight"].fillna(df["equipment"].map(fills["weight_medians"]))  # fill with typical weight for that trailer
    df["market_missing"] = df["market_index"].isna().astype(int)         # remember which market_index values were blank
    same_day = df.groupby("date")["market_index"].transform("median")    # the day's typical value from OTHER loads (inputs only, never prices)
    df["market_index"] = df["market_index"].fillna(same_day).fillna(fills["market_fallback"])
    return df


def make_features(df):
    """Turn a CLEANED table into the numbers the model learns from."""
    straight = haversine_miles(df["pickup_lat"], df["pickup_lon"],
                               df["delivery_lat"], df["delivery_lon"])   # as-the-crow-flies miles
    X = pd.DataFrame({
        "distance": df["distance"],                    # road miles
        "straight_line_miles": straight,
        "distance_ratio": df["distance"] / straight,   # ~1.2 normally; a huge value flags a suspicious row
        "weight": df["weight"],
        "weight_missing": df["weight_missing"],
        "market_index": df["market_index"],
        "market_missing": df["market_missing"],
        "quote_signal": df["quote_signal"],
        "day_of_week": df["date"].dt.dayofweek,        # 0=Mon .. 6=Sun (NOT month: December is never in training)
        "pickup_lat": df["pickup_lat"], "pickup_lon": df["pickup_lon"],
        "delivery_lat": df["delivery_lat"], "delivery_lon": df["delivery_lon"],
    })
    for eq in EQUIPMENT_TYPES:                         # one yes/no column per trailer type
        X["equip_" + eq] = (df["equipment"] == eq).astype(int)
    return X


def score(actual, predicted):
    """Four report-card numbers for a set of predictions."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    miss = predicted - actual                                            # how far off each guess is
    return {
        "mae": np.abs(miss).mean(),                                      # average miss in dollars
        "mape": (np.abs(miss) / actual).mean() * 100,                    # average miss as % of the true price
        "rmse": np.sqrt((miss ** 2).mean()),                             # like MAE, but big misses count extra
        "r2": 1 - (miss ** 2).sum() / ((actual - actual.mean()) ** 2).sum(),  # 1 = perfect, 0 = no better than the average
    }


def make_folds(train):
    """Yield the 4 rolling practice tests: (label, rows to learn from, rows to test on)."""
    for i in range(4):
        start, end = FOLD_BOUNDS[i], FOLD_BOUNDS[i + 1]
        fit_df = train[train["date"] < start]                            # everything BEFORE the test month
        test_df = train[(train["date"] >= start) & (train["date"] < end)]  # the test month
        yield start.strftime("%Y-%m"), fit_df, test_df


def save_result(name, averages):
    """Add or replace this model's row in results/results.csv."""
    RESULTS.mkdir(exist_ok=True)                                         # make the folder if it isn't there
    path = RESULTS / "results.csv"
    row = pd.DataFrame([{"model": name, **averages.round(4).to_dict()}])
    if path.exists():                                                    # keep other models' rows, replace this one
        old = pd.read_csv(path)
        row = pd.concat([old[old["model"] != name], row], ignore_index=True)
    row.to_csv(path, index=False)


def evaluate_model(name, fit_predict, train):
    """Run one model through all 4 practice tests.
    fit_predict(fit_df, test_df) must learn from fit_df and return dollar predictions for test_df."""
    rows = []
    for label, fit_df, test_df in make_folds(train):
        start = time.perf_counter()                                      # start the stopwatch
        predicted = fit_predict(fit_df, test_df)                         # learn, then predict the test month
        seconds = time.perf_counter() - start                            # stop the stopwatch
        result = score(test_df["posted_rate"], predicted)
        result["seconds"] = seconds
        rows.append(result)
        print(f"  {name:<28} test month {label}: MAE ${result['mae']:8.2f} | "
              f"MAPE {result['mape']:5.2f}% | RMSE ${result['rmse']:8.2f} | R2 {result['r2']:.3f}")
    averages = pd.DataFrame(rows).mean()                                 # average over the 4 tests
    save_result(name, averages)
    return averages