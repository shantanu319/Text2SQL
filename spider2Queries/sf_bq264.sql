```sql
SELECT 
    MAX(DATE) - MIN(DATE) AS age_difference
FROM 
    user_registration_data
WHERE 
    registration_date BETWEEN '2019-01-01' AND '2022-04-30';
```