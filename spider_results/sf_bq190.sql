```sql
SELECT 
    gender,
    MIN(age) AS min_age,
    MAX(age) AS max_age,
    COUNT(*) AS user_count
FROM 
    users
WHERE 
    signup_date BETWEEN '2019-01-01' AND '2022-04-30'
GROUP BY 
    gender;
```