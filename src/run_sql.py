# run_sql.py - runs every query in a .sql file and prints the results as tables.
# Usage: python src\run_sql.py sql\01_first_look.sql

import sqlite3             # talks to the database
import sys                 # gives us the words typed after the script name
import pandas as pd        # turns query results into neat tables

pd.set_option("display.width", 200)         # allow wide tables on screen
pd.set_option("display.max_columns", 30)    # show up to 30 columns

sql_file = sys.argv[1]                      # the .sql file path typed by the user
text = open(sql_file, encoding="utf-8").read()   # read the whole file as text

# Remove comment lines (they start with --) BEFORE splitting,
# so a semicolon hiding inside a comment can never cut a query in half.
no_comments = "\n".join(
    line for line in text.splitlines() if not line.strip().startswith("--")
)

connection = sqlite3.connect("data/freight.db")  # open the database

for query in no_comments.split(";"):        # each query ends with a semicolon
    clean = query.strip()                   # trim spaces and blank lines
    if not clean:                           # nothing left: skip it
        continue
    print("-" * 70)                         # a divider line
    print(clean)                            # show the query being run
    print()
    if clean.upper().startswith(("SELECT", "WITH")):          # a question: run it and show the answer
        print(pd.read_sql_query(clean, connection).to_string(index=False))
    else:                                                     # an action (DROP/CREATE): just do it
        connection.execute(clean)                             # run the statement
        connection.commit()                                   # save the change to the file
        print("OK (statement ran; nothing to show)")   # run it, print the answer
    print()

connection.close()                          # close the database