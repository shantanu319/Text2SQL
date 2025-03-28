```sql
SELECT COUNT(*) AS commit_count
FROM commits
JOIN repositories ON commits.repository_id = repositories.id
WHERE repositories.language = 'Shell'
  AND repositories.license = 'apache-2.0'
  AND LENGTH(commits.message) BETWEEN 6 AND 9999
  AND commits.message NOT LIKE '%merge%'
  AND commits.message NOT LIKE '%update%'
  AND commits.message NOT LIKE '%test%';
```