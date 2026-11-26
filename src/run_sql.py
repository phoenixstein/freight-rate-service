# run_sql.py - runs every query in a .sql file and prints the results as tables.
# Usage: python src\run_sql.py sql\01_first_look.sql

import sqlite3             # talks to the database
import sys                 # gives us the words typed after the script name
import pandas as pd        # turns query results into neat tables

pd.set_option("display.width", 200)         # allow wide tables on screen
pd.set_option("display.max_columns", 30)    # show up to 30 columns

sql_file = sys.argv[1]                      # the .sql file path typed by the user
text = open(sql_file, encoding="utf-8").read()   # read the whole file as text
connection = sqlite3.connect("data/freight.db")  # open the database

for query in text.split(";"):               # each query ends with a semicolon
    # Drop comment lines (they start with --) so we can see if anything is left.
    lines = [line for line in query.splitlines() if not line.strip().startswith("--")]
    clean = "\n".join(lines).strip()
    if not clean:                           # nothing but comments: skip it
        continue
    print("-" * 70)                         # a divider line
    print(clean)                            # show the query being run
    print()
    print(pd.read_sql_query(clean, connection).to_string(index=False))   # run it, print the answer
    print()

connection.close()                          # close the database