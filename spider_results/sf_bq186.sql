SELECT YEAR(trip_start_date) * 100 + MONTH(trip_start_date) AS year_month,
       MIN(duration_minutes),
       MAX(duration_minutes)
FROM bike_share_trips
GROUP BY YEAR(trip_start_date), MONTH(trip_start_date)
ORDER BY YEAR(trip_start_date), MONTH(trip_start_date);