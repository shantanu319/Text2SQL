```sql
SELECT 
    T2.state_abbr,
    COUNT(*) AS event_count
FROM 
    storm_events AS T1
JOIN 
    states AS T2 ON T1.state = T2.state_name
WHERE 
    T1.event_year BETWEEN 1980 AND 1995
GROUP BY 
    T2.state_abbr
ORDER BY 
    event_count DESC
LIMIT 5;
```