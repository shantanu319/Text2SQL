```sql
SELECT 
    quickplay_event_type,
    COUNT(CASE WHEN session_start BETWEEN '2018-07-02' AND '2018-07-16' THEN 1 ELSE NULL END) AS total_users,
    COUNT(CASE WHEN session_start BETWEEN '2018-07-02' AND '2018-07-16' AND quickplay_event_date >= DATEADD(day, -14, session_start) THEN 1 ELSE NULL END) AS retained_users,
    CAST(COUNT(CASE WHEN session_start BETWEEN '2018-07-02' AND '2018-07-16' AND quickplay_event_date >= DATEADD(day, -14, session_start) THEN 1 ELSE NULL END) AS REAL) / COUNT(CASE WHEN session_start BETWEEN '2018-07-02' AND '2018-07-16' THEN 1 ELSE NULL END) * 100 AS retention_rate
FROM 
    quickplay_events
WHERE 
    session_start BETWEEN '2018-07-02' AND '20