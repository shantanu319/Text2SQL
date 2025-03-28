```sql
SELECT 
    T3.start_station_name,
    ROUND(T4.avg_bike_duration, 3) AS avg_bike_duration
FROM 
    (SELECT 
         T1.route_id,
         COUNT(*) AS num_trips
     FROM 
         trips AS T1
     JOIN 
         citibike_routes AS T2 ON T1.route_id = T2.route_id
     WHERE 
         T1.year = 2016
     GROUP BY 
         T1.route_id
     ORDER BY 
         num_trips DESC
     LIMIT 20) AS T3
JOIN 
    citibike_routes AS T4 ON T3.route_id = T4.route_id
WHERE 
    T4.avg_bike_duration > (
        SELECT 
            AVG(avg_bike_duration)
        FROM 
            trips AS T5
        JOIN 
            citibike_routes AS T6 ON T5.route_id = T6.route_id
        WHERE 
            T5.year = 2016 AND T6.color = 'yellow'
    )
ORDER BY 
    T4.avg_bike_duration DESC
LIMIT 1;
```