```sql
SELECT 
    lat,
    lon,
    date,
    AVG(wind_speed) AS avg_wind_speed
FROM 
    icoads_core_data
WHERE 
    date BETWEEN '2005-01-01' AND '2015-12-31'
    AND wind_speed IS NOT NULL
GROUP BY 
    lat, lon, date
ORDER BY 
    avg_wind_speed DESC
LIMIT 5;
```