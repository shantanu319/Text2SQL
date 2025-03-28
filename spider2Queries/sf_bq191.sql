```sql
SELECT 
    T1.name AS repo_name,
    COUNT(DISTINCT T2.watcher_id) AS distinct_watcher_count
FROM 
    sample_files AS T1
JOIN 
    github_watchevent AS T2 ON T1.file_path = T2.repo_path
WHERE 
    YEAR(T2.created_at) = 2017
GROUP BY 
    T1.name
HAVING 
    COUNT(DISTINCT T2.watcher_id) > 300
ORDER BY 
    distinct_watcher_count DESC
LIMIT 2;
```