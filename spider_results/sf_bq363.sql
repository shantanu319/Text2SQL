SELECT 
    ROUND(trip_duration / 60, 2) AS duration_in_minutes,
    COUNT(*) AS num_trips,
    FORMAT(ROUND(AVG(fare), 2), '###.##') AS avg_fare_formatted
FROM 
    taxi_trips
WHERE 
    trip_duration BETWEEN 1 AND 50
GROUP BY 
    ROUND(trip_duration / 60, 2)
ORDER BY 
    duration_in_minutes