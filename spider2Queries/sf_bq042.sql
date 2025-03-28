```sql
SELECT 
    YEAR(date) AS Year,
    AVG(temperature) AS AverageTemperature,
    AVG(wind_speed) AS AverageWindSpeed,
    SUM(precipitation) AS TotalPrecipitation
FROM 
    weather_data
WHERE 
    station_id = 725030 AND 
    date BETWEEN '2011-06-12' AND '2020-06-12'
GROUP BY 
    YEAR(date);
```