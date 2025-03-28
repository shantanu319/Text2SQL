```sql
SELECT 
    YEAR(T.date) AS year,
    MONTH(T.date) AS month,
    SUM(ABS(T.air_temperature - T.wet_bulb_temperature)) +
    SUM(ABS(T.air_temperature - T.dew_point_temperature)) +
    SUM(ABS(T.air_temperature - T.sea_surface_temperature)) +
    SUM(ABS(T.wet_bulb_temperature - T.dew_point_temperature)) +
    SUM(ABS(T.wet_bulb_temperature - T.sea_surface_temperature)) +
    SUM(ABS(T.dew_point_temperature - T.sea_surface_temperature)) AS total_difference
FROM 
    temperatures AS T
WHERE 
    YEAR(T.date) BETWEEN 2010 AND 2014
GROUP BY 
    YEAR(T.date), MONTH(T.date)
ORDER BY 
    total_difference ASC
LIMIT 3;
```

This SQL query calculates the sum of absolute differences for each month between 2010 and 2014 across four different temperature variables (air temperature, wet bulb temperature, dew point temperature, and sea surface temperature). It then orders these sums in ascending order to find the months with the smallest differences. The result includes the year and month in numerical format along with the total