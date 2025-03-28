Certainly! To calculate the total number of sessions for each traffic channel in December 2020, you can use the following SQL query. This assumes that your table is named `event_params` and it contains columns such as `session_id`, `traffic_channel`, and `date`.

```sql
SELECT 
    traffic_channel,
    COUNT(session_id) AS total_sessions
FROM 
    event_params
WHERE 
    date BETWEEN '2020-12-01' AND '2020-12-31'
GROUP BY 
    traffic_channel;
```

This query selects the `traffic_channel` and counts the number of `session_id`s for each channel within the specified date range (December 1, 2020, to December 31, 2020). The results are grouped by `traffic_channel`, so you get a count of sessions for each unique traffic channel.