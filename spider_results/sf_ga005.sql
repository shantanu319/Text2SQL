```sql
SELECT 
    DATE_TRUNC('week', first_session) AS cohort_week,
    COUNT(DISTINCT user_id) AS total_users_in_cohort,
    SUM(CASE WHEN return_date IS NOT NULL THEN 1 ELSE 0 END) AS users_returned_in_week_2,
    (SUM(CASE WHEN return_date IS NOT NULL THEN 1 ELSE 0 END) * 100 / COUNT(DISTINCT user_id)) AS retention_rate
FROM (
    SELECT 
        user_id,
        MIN(session_start) AS first_session,
        MAX(session_start) AS last_session,
        CASE 
            WHEN MAX(session_start) - MIN(session_start) <= INTERVAL '2 weeks' THEN NULL
            ELSE MIN(session_start) + INTERVAL '2 weeks'
        END AS return_date
    FROM 
        sessions
    WHERE 
        session_start >= '2018-07-09'
    GROUP BY 
        user_id
) AS cohort_data
GROUP BY 
    cohort_week
ORDER BY 
    cohort_week;
```

This SQL query performs the following steps:
1. It filters the sessions table to include only those where the `session_start` date is after July 9, 2018