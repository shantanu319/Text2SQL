```sql
-- Step 1: Calculate the percentage change in the total number of unsheltered homeless people from 2015 to 2018 for each state

WITH StatePercentageChanges AS (
    SELECT 
        State,
        SUM(CASE WHEN Year = 2015 THEN Count ELSE 0 END) AS TotalCount_2015,
        SUM(CASE WHEN Year = 2018 THEN Count ELSE 0 END) AS TotalCount_2018,
        ((SUM(CASE WHEN Year = 2018 THEN Count ELSE 0 END) - SUM(CASE WHEN Year = 2015 THEN Count ELSE 0 END)) / SUM(CASE WHEN Year IN (2015, 2018) THEN Count ELSE 0 END)) * 100 AS PercentageChange
    FROM 
        UnshelteredHomelessData
    GROUP BY 
        State
)

-- Step 2: Determine the national average of these state percentage changes

SELECT AVG(PercentageChange) AS NationalAveragePercentageChange
FROM 
    StatePercentageChanges;

-- Step 3: Identify the five states whose percentage change is closest to this national average percentage change