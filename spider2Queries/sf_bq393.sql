```sql
SELECT 
    id,
    month_number
FROM 
    users
WHERE 
    status = 'inactive'
    AND last_activity_month < (
        SELECT 
            MAX(last_activity_month)
        FROM 
            users
        WHERE 
            last_activity_month <= '2024-09-10'
    )
ORDER BY 
    month_number DESC
LIMIT 1;
```