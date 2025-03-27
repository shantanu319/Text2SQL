```sql
SELECT AVG(engaged_sessions) AS avg_engaged_sessions
FROM (
    SELECT COUNT(DISTINCT ga_session_id) AS engaged_sessions
    FROM your_table_name
    WHERE DATE_TRUNC('month', timestamp_column) = '2020-12'
      AND event_parameter = 'session_engaged'
      AND user_pseudo_id IS NOT NULL
) AS session_counts;
```

Replace `your_table_name` with the actual name of your table, and `timestamp_column` with the column that contains the timestamp information. This query calculates the average number of engaged sessions per user in December 2020, considering only sessions where the event parameter 'session_engaged' is set to '1'. It uses `COUNT(DISTINCT ga_session_id)` to ensure each session is counted only once per user. The `DATE_TRUNC` function is used to filter sessions by month. Ensure that the data types and column names match those in your database schema.