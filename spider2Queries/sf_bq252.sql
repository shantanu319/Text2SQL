```sql
SELECT r.name AS repository_name
FROM repositories r
JOIN files f ON r.id = f.repository_id
WHERE f.type != 'binary'
GROUP BY r.id, r.name
ORDER BY COUNT(DISTINCT f.id) DESC
LIMIT 1;
```