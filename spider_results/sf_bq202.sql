```sql
SELECT 
    DAYOFWEEK(start_time) AS day_of_week,
    HOUR(start_time) AS hour_of_day,
    COUNT(*) AS trip_count
FROM 
    citibike_trips
WHERE 
    YEAR(start_time) = 2018
GROUP BY 
    DAYOFWEEK(start_time), HOUR(start_time)
ORDER BY 
    trip_count DESC
LIMIT 1;
```