```sql
SELECT COUNT(DISTINCT session_id) AS unique_sessions
FROM (
    SELECT session_id,
           MIN(page_path) AS first_page,
           MAX(page_path) AS last_page
    FROM page_views
    WHERE page_path IN ('/regist/input', '/regist/confirm')
    GROUP BY session_id
) AS filtered_sessions
WHERE first_page = '/regist/input' AND last_page = '/regist/confirm';
```