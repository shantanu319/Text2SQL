```sql
SELECT 
    age,
    SUM(CASE WHEN race_description LIKE '%black%' THEN 1 ELSE 0 END) AS black_deaths,
    COUNT(*) AS total_deaths
FROM 
    EntityAxisConditions
JOIN 
    DeathRecords ON EntityAxisConditions.death_id = DeathRecords.id
WHERE 
    age BETWEEN 12 AND 18
    AND ICD_10_code_description LIKE '%vehicle%'
    OR ICD_10_code_description LIKE '%firearm%'
GROUP BY 
    age;
```