```sql
SELECT 
    station_id,
    COUNT(*) AS total_starting_trips,
    CAST(COUNT(*) AS REAL) * 100 / SUM(COUNT(*)) OVER () AS percentage_of_total,
    AVG(trip_duration_minutes) AS average_trip_duration
FROM 
    trips
WHERE 
    station_id IN (
        SELECT 
            station_id
        FROM 
            stations
        WHERE 
            status = 'active'
    )
GROUP BY 
    station_id
ORDER BY 
    RANK() OVER (PARTITION BY station_id ORDER BY total_starting_trips DESC);
```

This SQL query selects the necessary columns from the `trips` table, filters for active stations, groups by station ID, calculates the total number of starting trips, the percentage of those trips relative to the total starting trips from active stations, and the average trip duration. It then orders the results by the station's rank within its group of active stations. The `RANK()` function is used to determine the rank of each station based on the total number of starting trips.