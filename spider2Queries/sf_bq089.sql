```sql
SELECT 
    COUNT(*) / (SUM(CAST(TotalPop AS REAL)) / 1000) * 1000 AS VaccineSitesPer1000People
FROM 
    Counties
WHERE 
    State = 'California';
```