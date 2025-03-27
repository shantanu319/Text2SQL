To achieve the desired result using SQL, you can follow these steps:

1. Calculate the average runs per match for each player.
2. Group the results by country and calculate the average of these player averages for each country.

Here's a sample SQL query that accomplishes this:

```sql
WITH PlayerAverageRuns AS (
    SELECT 
        Country,
        PlayerID,
        AVG(RunsPerMatch) AS AverageRunsPerMatch
    FROM 
        Players
    GROUP BY 
        Country, PlayerID
),
CountryBattingAverages AS (
    SELECT 
        Country,
        AVG(AverageRunsPerMatch) AS CountryBattingAverage
    FROM 
        PlayerAverageRuns
    GROUP BY 
        Country
)
SELECT 
    Country,
    CountryBattingAverage
FROM 
    CountryBattingAverages
ORDER BY 
    CountryBattingAverage DESC
LIMIT 5;
```

### Explanation:

- **PlayerAverageRuns CTE**: This Common Table Expression (CTE) calculates the average runs per match for each player. It groups the data by `Country` and `PlayerID`, and computes the average runs per match for each player.

- **CountryBattingAverages CTE**: This CTE calculates the average of the player