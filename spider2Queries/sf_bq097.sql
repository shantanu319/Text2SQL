```sql
SELECT 
    GeoName, 
    AVG(EarningsPerJob) AS AverageEarningsPerJob, 
    LAG(AVG(EarningsPerJob)) OVER (PARTITION BY GeoName ORDER BY Year) AS PreviousYearAverage,
    AVG(EarningsPerJob) - LAG(AVG(EarningsPerJob)) OVER (PARTITION BY GeoName ORDER BY Year) AS IncreaseInAverageEarnings
FROM 
    EarningsData
WHERE 
    GeoName LIKE '%MA' AND Year BETWEEN 2012 AND 2017
GROUP BY 
    GeoName, Year;
```