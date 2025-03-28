To convert the given text into an SQL query, we need to break down the requirements and construct a query that performs the specified analysis. Here's how you can do it:

```sql
WITH UserSessions AS (
    SELECT 
        user_id,
        event_date,
        DATE_TRUNC('week', event_date) AS week_start
    FROM 
        your_events_table
    WHERE 
        event_date >= '2018-07-02' AND 
        event_date <= '2018-10-02'
),
FirstSessionWeeks AS (
    SELECT 
        user_id,
        MIN(week_start) AS first_session_week
    FROM 
        UserSessions
    GROUP BY 
        user_id
),
WeeklyRetention AS (
    SELECT 
        fs.user_id,
        fs.first_session_week,
        ws.week_start,
        COUNT(DISTINCT CASE WHEN ws.event_date = fs.first_session_week THEN ws.user_id ELSE NULL END) AS new_users_in_week_0,
        COUNT(DISTINCT CASE WHEN ws.event_date BETWEEN fs.first_session_week + INTERVAL '1 week' AND fs.first_session_week + INTERVAL '5 weeks' THEN ws.user_id ELSE NULL END) AS retained_users
    FROM 
        First