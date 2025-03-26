SELECT browser_name, AVG(session_duration) AS avg_session_duration FROM (
    SELECT 
        b.browser_name,
        MIN(s.session_timestamp) - MAX(s.session_timestamp) AS session_duration
    FROM 
        Sessions s
    JOIN 
        Browsers b ON s.user_id = b.user_id
    GROUP BY 
        b.browser_name
    HAVING 
        COUNT(*) > 10
)