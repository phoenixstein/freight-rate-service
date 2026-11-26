-- One row per city with its coordinates, collected from every pickup and delivery in both tables.
DROP TABLE IF EXISTS cities;

CREATE TABLE cities AS
SELECT city,
       AVG(lat) AS lat,
       AVG(lon) AS lon,
       MAX(lat) - MIN(lat) AS lat_spread,
       COUNT(*) AS seen
FROM (
    SELECT pickup AS city, pickup_lat AS lat, pickup_lon AS lon FROM train_loads
    UNION ALL
    SELECT delivery, delivery_lat, delivery_lon FROM train_loads
    UNION ALL
    SELECT pickup, pickup_lat, pickup_lon FROM validation_loads
    UNION ALL
    SELECT delivery, delivery_lat, delivery_lon FROM validation_loads
)
GROUP BY city
ORDER BY city;

-- How many cities, and does any city have two different coordinates (spread should be 0)?
SELECT COUNT(*) AS cities, MAX(lat_spread) AS worst_lat_spread FROM cities;

SELECT * FROM cities ORDER BY seen DESC LIMIT 5;