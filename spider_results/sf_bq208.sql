```sql
SELECT 
    ws.station_id,
    COUNT(*) AS num_valid_temps
FROM 
    weather_stations ws
JOIN 
    observations obs ON ws.station_id = obs.station_id
WHERE 
    ws.latitude BETWEEN 41.197 - 0.2 AND 41.197 + 0.2
    AND ws.longitude BETWEEN -73.764 - 0.2 AND -73.764 + 0.2
    AND YEAR(obs.observation_date) BETWEEN 2011 AND 2020
    AND obs.temperature IS NOT NULL
GROUP BY 
    ws.station_id;
```