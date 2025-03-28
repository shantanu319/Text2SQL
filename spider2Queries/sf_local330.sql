```sql
SELECT 
    web_page,
    COUNT(DISTINCT user_session) AS unique_user_sessions
FROM 
    log_table
WHERE 
    (timestamp = MIN(timestamp) OR timestamp = MAX(timestamp))
GROUP BY 
    web_page;
```