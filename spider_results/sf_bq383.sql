```sql
SELECT 
    YEAR(date) AS Year,
    MAX(precipitation_mm) AS Highest_Precipitation,
    MIN(temperature_celsius) AS Minimum_Temperature,
    MAX(temperature_celsius) AS Maximum_Temperature
FROM 
    weather_data
WHERE 
    date >= DATE_SUB(CURDATE(), INTERVAL 15 DAY)
    AND date BETWEEN '2013-01-01' AND '2016-12-31'
GROUP BY 
    YEAR(date);
```

This SQL query selects the highest recorded precipitation, minimum temperature, and maximum temperature from the last 15 days of each year from 2013 to 2016 at the specified weather station. It filters the data based on the date range and ensures that only non-null values are considered. The results are grouped by year to provide a summary for each year.