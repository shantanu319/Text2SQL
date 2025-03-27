```sql
SELECT YEAR
FROM (
    SELECT YEAR,
           SUM(CASE WHEN POSITION = 1 THEN POINTS ELSE 0 END) AS DriverPoints,
           SUM(CASE WHEN POSITION = 1 THEN 0 ELSE POINTS END) AS ConstructorPoints
    FROM results
    GROUP BY YEAR
) AS SubQuery
ORDER BY DriverPoints + ConstructorPoints ASC
LIMIT 3;
```