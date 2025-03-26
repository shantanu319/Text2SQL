SELECT 
    ROUND(MIN(duration) / 60, 0) AS min_duration,
    ROUND(MAX(duration) / 60, 0) AS max_duration,
    COUNT(*) AS num_trips,
    AVG(fare) AS avg_fare