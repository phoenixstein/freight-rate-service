# check_sql_vs_pandas.py - proves the SQL daily_stats table matches plain pandas.

import sqlite3             # opens the database
import pandas as pd        # recomputes the same numbers the pandas way

connection = sqlite3.connect("data/freight.db")                        # open the database
sql_stats = pd.read_sql_query("SELECT * FROM daily_stats", connection)  # the SQL answer
connection.close()                                                      # done with the database

problems = 0                                                            # counts mismatches
for source, csv_name in [("train", "train_test.csv"), ("validation", "validation.csv")]:
    df = pd.read_csv("data/" + csv_name)                                # original CSV
    # The pandas way: group by day, then count loads and take medians.
    pandas_stats = df.groupby("date").agg(
        loads=("load_id", "size"),
        quote_median=("quote_signal", "median"),
        market_index_median=("market_index", "median"),
    ).reset_index()
    sql_part = sql_stats[sql_stats["source"] == source]                 # SQL rows for this source
    merged = sql_part.merge(pandas_stats, on="date", suffixes=("_sql", "_pandas"))   # line up by day
    print(source, "| days in SQL:", len(sql_part), "| days in pandas:", len(pandas_stats),
          "| days matched:", len(merged))
    for col in ["loads", "quote_median", "market_index_median"]:
        diff = (merged[col + "_sql"] - merged[col + "_pandas"]).abs().max()   # biggest gap
        print(f"  {col}: biggest difference = {diff}")
        if not (diff < 1e-9):                                           # not (almost) zero
            problems += 1

print("MATCH" if problems == 0 else f"{problems} PROBLEM(S)")           # final verdict