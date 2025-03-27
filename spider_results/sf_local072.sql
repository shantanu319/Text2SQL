```sql
WITH InsertedDays AS (
    SELECT 
        Country,
        DATE_TRUNC('day', InsertDate) AS InsertDay
    FROM 
        DataInsertions
    WHERE 
        YEAR(InsertDate) = 2022 AND MONTH(InsertDate) = 1
),
DistinctDays AS (
    SELECT DISTINCT 
        Country, 
        InsertDay
    FROM 
        InsertedDays
),
CountryWithNineDays AS (
    SELECT 
        Country
    FROM 
        DistinctDays
    GROUP BY 
        Country
    HAVING 
        COUNT(DISTINCT InsertDay) = 9
),
LongestConsecutivePeriod AS (
    SELECT 
        Country,
        InsertDay,
        ROW_NUMBER() OVER (PARTITION BY Country ORDER BY InsertDay) - ROW_NUMBER() OVER (PARTITION BY Country ORDER BY InsertDay DESC) AS ConsecutiveGroup
    FROM 
        InsertedDays
)
SELECT 
    T1.Country,
    MAX(T2.ConsecutiveGroup) AS LongestConsecutivePeriod,
    SUM(CASE WHEN T3.CapitalCity = 'Yes' THEN 1 ELSE 0 END) / COUNT(*) AS CapitalCityProportion
FROM 
    CountryWithNineDays AS T1