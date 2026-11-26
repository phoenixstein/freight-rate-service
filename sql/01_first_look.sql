-- 1. How many loads are in each table?
SELECT 'train_loads' AS table_name, COUNT(*) AS loads FROM train_loads
UNION ALL
SELECT 'validation_loads', COUNT(*) FROM validation_loads;

-- 2. Loads and prices per month (substr takes the first 7 characters of the date: YYYY-MM).
-- Multiplying by 1.0 forces decimal division instead of whole-number division.
SELECT substr(date, 1, 7)                        AS month,
       COUNT(*)                                  AS loads,
       ROUND(AVG(posted_rate), 0)                AS avg_rate,
       ROUND(AVG(posted_rate * 1.0 / distance), 3) AS avg_rate_per_mile
FROM train_loads
GROUP BY month
ORDER BY month;

-- 3. The same by trailer type.
SELECT equipment,
       COUNT(*)                         AS loads,
       ROUND(AVG(posted_rate), 0)       AS avg_rate
FROM train_loads
GROUP BY equipment
ORDER BY loads DESC;

-- 4. Spike loads per month (ratio = real price / (quote x distance); up > 2, down < 0.6).
SELECT substr(date, 1, 7) AS month,
       COUNT(*) AS loads,
       SUM(CASE WHEN posted_rate * 1.0 / (quote_signal * distance) > 2   THEN 1 ELSE 0 END) AS spike_up,
       SUM(CASE WHEN posted_rate * 1.0 / (quote_signal * distance) < 0.6 THEN 1 ELSE 0 END) AS spike_down
FROM train_loads
GROUP BY month
ORDER BY month;