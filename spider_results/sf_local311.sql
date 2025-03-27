```sql
SELECT 
    c.ConstructorName,
    y.Year,
    SUM(d.Points) AS TotalPoints
FROM 
    Constructors c
JOIN 
    Races r ON c.ConstructorId = r.ConstructorId
JOIN 
    Drivers d ON r.DriverId = d.DriverId
JOIN 
    Teams t ON d.TeamId = t.TeamId
JOIN 
    Years y ON r.Year = y.Year
WHERE 
    (d.DriverId, d.TeamId) IN (
        SELECT 
            DriverId, TeamId
        FROM 
            Drivers
        ORDER BY 
            Points DESC
        LIMIT 1
    )
GROUP BY 
    c.ConstructorName, y.Year
ORDER BY 
    TotalPoints DESC
LIMIT 3;
```