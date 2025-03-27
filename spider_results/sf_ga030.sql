```sql
SELECT 
    DATE_FORMAT(CAST(first_session_date AS DATE), '%Y-%m-%d') AS cohort_name,
    COUNT(DISTINCT user_id) AS total_users,
    SUM(CASE WHEN return_date IS NOT NULL THEN 1 ELSE 0 END) / COUNT(DISTINCT user_id) * 100 AS retention_rate
FROM (
    SELECT 
        user_id,
        MIN(session_date) AS first_session_date,
        MAX(return_date) AS last_return_date
    FROM 
        sessions
    WHERE 
        session_date >= '2018-07-02'
    GROUP BY 
        user_id
) AS user_sessions
LEFT JOIN (
    SELECT 
        user_id,
        session_date
    FROM 
        sessions
    WHERE 
        session_date >= '2018-07-02' + INTERVAL 3 WEEK
) AS fourth_week_sessions ON user_sessions.user_id = fourth_week_sessions.user_id
WHERE 
    fourth_week_sessions.session_date IS NOT NULL
GROUP BY 
    cohort_name
ORDER BY 
    retention_rate DESC
LIMIT 1;
```