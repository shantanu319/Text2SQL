```sql
SELECT 
    DATE_FORMAT(start_date, '%Y%m') AS year_month,
    MIN(duration) AS first_trip_duration,
    MAX(duration) AS last_trip_duration,
    MAX(duration) AS highest_trip_duration,
    MIN(duration) AS lowest_trip_duration
FROM 
    bike_share_trips
GROUP BY 
    DATE_FORMAT(start_date, '%Y%m')
ORDER BY 
    DATE_FORMAT(start_date, '%Y%m');
```