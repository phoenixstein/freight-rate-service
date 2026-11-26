"""
check_setup.py - Phase 1 health check.
Goal: confirm all four data files exist and look the way we expect,
BEFORE we build anything. It only reads files. It changes nothing.
"""
import pandas as pd
from pathlib import Path

DATA = Path("data")  # the folder that holds the CSV files

# What we EXPECT: file name -> (number of rows, number of columns)
EXPECTED = {
    "train_test.csv": (48000, 14),
    "validation.csv": (12000, 13),
    "validation_predictions_template.csv": (12000, 2),
    "december_chart_inputs.csv": (31, 7),
}

frames = {}      # we keep each loaded table here so we can reuse it below
all_ok = True    # turns False if anything looks wrong

for name, (rows, cols) in EXPECTED.items():
    path = DATA / name
    if not path.exists():                       # is the file even there?
        print(f"FAIL  {name}: file not found")
        all_ok = False
        continue
    df = pd.read_csv(path)                      # load the CSV into a table
    frames[name] = df
    ok = df.shape == (rows, cols)               # does the size match?
    all_ok = all_ok and ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}: shape {df.shape}, expected {(rows, cols)}")

if "train_test.csv" in frames and "validation.csv" in frames:
    train, val = frames["train_test.csv"], frames["validation.csv"]
    # Dates are stored as text, so convert them before asking for the earliest/latest.
    print("\nTrain dates:     ", pd.to_datetime(train["date"]).min().date(),
          "to", pd.to_datetime(train["date"]).max().date())
    print("Validation dates:", pd.to_datetime(val["date"]).min().date(),
          "to", pd.to_datetime(val["date"]).max().date())
    # Count blank cells per column (only shows columns that have blanks).
    for label, df in [("train", train), ("validation", val)]:
        blanks = df.isna().sum()
        print(f"Blank cells in {label}:", blanks[blanks > 0].to_dict())

if "validation_predictions_template.csv" in frames and "validation.csv" in frames:
    tmpl, val = frames["validation_predictions_template.csv"], frames["validation.csv"]
    # Every load we must predict has to appear in the template, with the same IDs.
    same_ids = set(tmpl["load_id"]) == set(val["load_id"])
    all_ok = all_ok and same_ids
    print(f"{'PASS' if same_ids else 'FAIL'}  template IDs match validation IDs")
    print("Empty predicted_rate cells in template:", int(tmpl["predicted_rate"].isna().sum()))

print("\nOVERALL:", "ALL CHECKS PASSED" if all_ok else "SOMETHING NEEDS FIXING")