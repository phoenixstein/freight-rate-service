-- Builds one row per (source, day): number of loads and the day's median quote and market index.

DROP TABLE IF EXISTS daily_stats;

CREATE TABLE daily_stats AS
WITH all_loads AS (
    -- Stack train and validation loads into one list, labelled by source.
    SELECT 'train' AS source, date, quote_signal, market_index FROM train_loads
    UNION ALL
    SELECT 'validation' AS source, date, quote_signal, market_index FROM validation_loads
),
quote_ranked AS (
    -- Number the quotes in order within each (source, day), and count how many there are.
    SELECT source, date, quote_signal AS value,
           ROW_NUMBER() OVER (PARTITION BY source, date ORDER BY quote_signal) AS rn,
           COUNT(*)     OVER (PARTITION BY source, date)                       AS n
    FROM all_loads
    WHERE quote_signal IS NOT NULL
),
quote_med AS (
    -- Keep the middle one or two rows and average them: that is the median.
    SELECT source, date, AVG(value) AS quote_median
    FROM quote_ranked
    WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
    GROUP BY source, date
),
market_ranked AS (
    SELECT source, date, market_index AS value,
           ROW_NUMBER() OVER (PARTITION BY source, date ORDER BY market_index) AS rn,
           COUNT(*)     OVER (PARTITION BY source, date)                        AS n
    FROM all_loads
    WHERE market_index IS NOT NULL
),
market_med AS (
    SELECT source, date, AVG(value) AS market_index_median
    FROM market_ranked
    WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
    GROUP BY source, date
),
day_counts AS (
    -- How many loads each day.
    SELECT source, date, COUNT(*) AS loads
    FROM all_loads
    GROUP BY source, date
)
SELECT d.source, d.date, d.loads, q.quote_median, m.market_index_median
FROM day_counts d
LEFT JOIN quote_med  q ON q.source = d.source AND q.date = d.date
LEFT JOIN market_med m ON m.source = d.source AND m.date = d.date
ORDER BY d.source, d.date;

-- Peek at the first rows of the validation days.
SELECT * FROM daily_stats WHERE source = 'validation' ORDER BY date LIMIT 5;