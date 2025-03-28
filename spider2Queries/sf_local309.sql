```sql
SELECT 
    YEAR,
    DriverFullName,
    ConstructorName,
    MAX(Points) AS MaxPoints
FROM 
    races
JOIN 
    drivers ON races.DriverID = drivers.DriverID
JOIN 
    constructors ON races.ConstructorID = constructors.ConstructorID
GROUP BY 
    YEAR, DriverFullName, ConstructorName;
```