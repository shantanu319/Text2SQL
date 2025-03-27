```sql
SELECT COUNT(DISTINCT user_id) AS distinct_users
FROM page_views
WHERE view_date BETWEEN '2021-01-01' AND '2021-01-31'
GROUP BY page_url
ORDER BY COUNT(*) DESC
LIMIT 1;
```