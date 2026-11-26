# build_database.py - copies the two CSV files into one SQLite database file.

import sqlite3                 # built-in Python tool for SQLite databases
from pathlib import Path       # safe handling of file paths
import pandas as pd            # reads the CSV files

DATA = Path("data")                    # the data folder (ignored by Git)
DB_FILE = DATA / "freight.db"          # the database file we will create

# Pairs of (CSV file, table name to create).
SOURCES = [("train_test.csv", "train_loads"), ("validation.csv", "validation_loads")]


def main():
    connection = sqlite3.connect(DB_FILE)             # open the database (creates it if missing)
    for csv_name, table_name in SOURCES:              # repeat for each pair
        df = pd.read_csv(DATA / csv_name)             # read the CSV into a pandas table
        # Copy it into the database. "replace" means a rerun rebuilds the table cleanly.
        df.to_sql(table_name, connection, if_exists="replace", index=False)
        # Ask the database how many rows it now holds, as a check.
        count = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"{table_name}: {count} rows")
    connection.close()                                # always close the database when done


if __name__ == "__main__":     # run main() only when this file is run directly
    main()