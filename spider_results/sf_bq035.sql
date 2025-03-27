```sql
SELECT 
    b.bike_id,
    SUM(t.distance) AS total_distance_traveled
FROM 
    trips t
JOIN 
    bikes b ON t.bike_id = b.id
WHERE 
    t.start_station_name LIKE '%San Francisco%'
GROUP BY 
    b.bike_id;
```