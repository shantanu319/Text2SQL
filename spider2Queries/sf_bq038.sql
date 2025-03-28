```sql
SELECT 
    station_name,
    COUNT(*) AS total_trips,
    SUM(CASE WHEN start_time = end_time THEN 1 ELSE 0 END) AS group_rides,
    (SUM(CASE WHEN start_time = end_time THEN 1 ELSE 0 END) * 1.0 / COUNT(*)) AS proportion_of_group_rides
FROM 
    citibike_trips
GROUP BY 
    station_name
ORDER BY 
    proportion_of_group_rides DESC
LIMIT 10;
```

This SQL query calculates the proportion of group rides for each Citibike station and identifies the top 10 stations with the highest proportions. It uses a `CASE` statement to count only those trips that start and end at the same station, indicating a group ride. The result is ordered in descending order based on the proportion of group rides and limited to the top 10 stations.