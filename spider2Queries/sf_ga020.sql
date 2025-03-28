```sql
SELECT 
    event_type,
    AVG(user_retention_rate) AS average_user_retention_rate
FROM 
    quickplay_events
WHERE 
    session_start BETWEEN '2018-08-01' AND '2018-08-15'
GROUP BY 
    event_type
ORDER BY 
    average_user_retention_rate ASC
LIMIT 1;
```