```sql
SELECT 
    gender,
    COUNT(*) AS total_users,
    MIN(age) AS youngest_age,
    MAX(age) AS oldest_age
FROM 
    users
WHERE 
    birthdate BETWEEN '2019-01-01' AND '2022-04-30'
GROUP BY 
    gender;
```