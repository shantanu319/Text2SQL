```sql
SELECT u.user_id, u.username
FROM users u
JOIN (
    SELECT p.collection_id, SUM(p.clicks) AS total_clicks
    FROM photos p
    GROUP BY p.collection_id
) pc ON u.user_id = pc.collection_id
ORDER BY pc.total_clicks DESC
LIMIT 1;
```