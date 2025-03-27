```sql
SELECT 
    YEAR,
    CAST(SUM(CASE WHEN gender = 'Female' THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS proportion_female_only,
    COUNT(*) AS total_films
FROM 
    films
WHERE 
    gender IN ('Male', 'None')
GROUP BY 
    YEAR;
```