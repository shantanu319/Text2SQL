```sql
SELECT DATE_FORMAT(T1.date, '%Y-%m-%d') AS date
FROM (
    SELECT date, COUNT(*) AS z_score
    FROM public_incidents
    WHERE incident_type = 'PUBLIC INTOXICATION'
      AND city = 'Austin'
      AND YEAR(date) = 2016
    GROUP BY date
    ORDER BY z_score DESC
    LIMIT 1 OFFSET 1
) T1;
```