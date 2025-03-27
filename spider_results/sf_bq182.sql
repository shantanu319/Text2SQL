```sql
SELECT T1.language AS primary_language
FROM (
    SELECT language, SUM(bytes) AS total_bytes
    FROM repository
    WHERE created_at >= '2023-01-18'
    GROUP BY language
    ORDER BY total_bytes DESC
    LIMIT 1
) AS T1
JOIN pull_request_event ON T1.language = pull_request_event.repository_id
WHERE pull_request_event.created_at >= '2023-01-18'
GROUP BY T1.language
HAVING COUNT(*) >= 100;
```