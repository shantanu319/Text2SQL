```sql
SELECT 
    app_group_id,
    DATE(start_day) AS start_day,
    COUNT(DISTINCT user_id) AS unique_users
FROM 
    sessions
WHERE 
    start_day BETWEEN '2023-06-01' AND '2023-06-07'
GROUP BY 
    app_group_id, 
    DATE(start_day);
```