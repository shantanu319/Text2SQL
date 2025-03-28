```sql
SELECT repo_name, COUNT(*) AS commit_count
FROM repositories
WHERE language = 'JavaScript'
GROUP BY repo_name
ORDER BY commit_count DESC
LIMIT 2;
```