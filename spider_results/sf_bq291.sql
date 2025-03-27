```sql
SELECT 
    DATE(forecast_date) AS forecast_date,
    MAX(temperature) AS max_temperature,
    MIN(temperature) AS min_temperature,
    AVG(temperature) AS avg_temperature,
    SUM(precipitation) AS total_precipitation,
    AVG(CASE WHEN time BETWEEN '10:00' AND '17:00' THEN cloud_cover ELSE NULL END) AS avg_cloud_cover_10am_to_5pm,
    SUM(CASE WHEN temperature < 32 THEN snowfall ELSE NULL END) AS total_snowfall,
    SUM(CASE WHEN temperature >= 32 THEN rainfall ELSE NULL END) AS total_rainfall
FROM 
    weather_forecasts
WHERE 
    latitude = 26.75
    AND longitude = 51.5
    AND ST_DWithin(location, ST_MakePoint(26.75, 51.5), 5000)
    AND YEAR(forecast_date) = 2019
    AND MONTH(forecast_date) = 7
GROUP BY 
    forecast_date;
```

This SQL query retrieves the required weather summary for each forecast date within the specified conditions. It