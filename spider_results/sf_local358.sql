```sql
SELECT 
    CASE 
        WHEN age BETWEEN 20 AND 29 THEN '20s'
        WHEN age BETWEEN 30 AND 39 THEN '30s'
        WHEN age BETWEEN 40 AND 49 THEN '40s'
        WHEN age BETWEEN 50 AND 59 THEN '50s'
        ELSE 'Others'
    END AS age_category,
    COUNT(*) AS user_count
FROM 
    users
GROUP BY 
    age_category;
```