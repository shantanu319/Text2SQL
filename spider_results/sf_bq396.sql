```sql
SELECT 
    State,
    MAX(Total_Rainy) - MIN(Total_Clear) AS Difference,
    Total_Rainy,
    Total_Clear
FROM (
    SELECT 
        State,
        SUM(CASE WHEN Weather = 'Rainy' AND DayOfWeek IN (5, 6) THEN 1 ELSE 0 END) AS Total_Rainy,
        SUM(CASE WHEN Weather = 'Clear' AND DayOfWeek IN (5, 6) THEN 1 ELSE 0 END) AS Total_Clear
    FROM TrafficAccidents
    WHERE Year = 2016
    GROUP BY State
) AS SubQuery
ORDER BY Difference DESC
LIMIT 3;
```