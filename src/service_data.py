# service_data.py - looks things up in the SQLite database for the API.

import sqlite3                    # talks to the database
from contextlib import closing    # makes sure a connection is closed after use
from pathlib import Path          # safe file paths

DB_FILE = Path("data/freight.db")     # the database built by build_database.py


def get_city(name):
    """Return (lat, lon) for a city name (ignoring upper/lower case), or None if unknown."""
    with closing(sqlite3.connect(DB_FILE)) as conn:              # open, and always close afterwards
        row = conn.execute(
            "SELECT lat, lon FROM cities WHERE lower(city) = lower(?)",   # ? = safe placeholder
            (name.strip(),),                                      # the value that fills the ?
        ).fetchone()                                              # first matching row, or None
    return row


def get_day_stats(day):
    """Return (typical quote, typical market index) for a 'YYYY-MM-DD' day, or None if unknown."""
    with closing(sqlite3.connect(DB_FILE)) as conn:
        row = conn.execute(
            "SELECT quote_median, market_index_median FROM daily_stats WHERE date = ?",
            (day,),
        ).fetchone()
    if row is None or row[0] is None or row[1] is None:           # unknown day or blank medians
        return None
    return row