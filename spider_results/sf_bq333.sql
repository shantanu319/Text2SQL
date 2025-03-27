```sql
SELECT 
    browser_name,
    AVG(TIMESTAMPDIFF(SECOND, MIN(timestamp), MAX(timestamp))) AS average_session_duration
FROM 
    sessions
WHERE 
    COUNT(*) > 10
GROUP BY 
    browser_name
ORDER BY 
    average_session_duration ASC
LIMIT 3;
```