```sql
SELECT 
    ConstructorName,
    COUNT(*) AS FewestPointsSeasons
FROM (
    SELECT 
        DriverID,
        Season,
        SUM(Points) AS TotalPoints
    FROM 
        F1Drivers
    WHERE 
        Season >= 2001 AND Points > 0
    GROUP BY 
        DriverID, Season
) AS PointScorers
JOIN 
    Constructors ON PointScorers.DriverID = Drivers.DriverID
GROUP BY 
    ConstructorName
ORDER BY 
    FewestPointsSeasons DESC
LIMIT 5;
```

This query calculates the number of seasons each constructor has had where their drivers scored the fewest total points among all point-scoring drivers in that season. It then selects the top 5 constructors with the highest count of such seasons. Note that this assumes there is a `Constructors` table and a `F1Drivers` table with relevant columns for driver ID, season, and points. Adjust the table and column names as necessary to fit your database schema.