```sql
SELECT 
    T2.language,
    COUNT(T1.path) AS file_count
FROM 
    github_repos.sample_files AS T1
JOIN 
    github_repos.sample_contents AS T2 ON T1.id = T2.file_id
WHERE 
    T2.content IS NOT NULL AND T2.content <> ''
GROUP BY 
    T2.language
ORDER BY 
    file_count DESC
LIMIT 10;
```