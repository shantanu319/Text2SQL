```sql
SELECT 
    COUNT(CASE WHEN event_type = 'page_view' AND page_name IN ('Home', 'Checkout Confirmation') THEN 1 ELSE NULL END) AS total_conversions,
    COUNT(CASE WHEN event_type = 'page_view' AND page_name = 'Home' THEN 1 ELSE NULL END) AS total_visits,
    (total_conversions / total_visits) * 100 AS conversion_rate
FROM 
    user_sessions
WHERE 
    session_date = '2021-01-02'
AND 
    event_type = 'page_view';
```