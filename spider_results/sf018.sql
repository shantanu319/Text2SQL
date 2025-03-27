```sql
SELECT 
    user_id,
    COUNT(*) AS notification_count
FROM 
    push_notifications
WHERE 
    created_at >= '2023-06-01 00:00:00'
    AND created_at < '2023-06-01 01:00:00'
GROUP BY 
    user_id;
```