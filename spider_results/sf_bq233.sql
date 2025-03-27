```sql
SELECT 
    CASE 
        WHEN file_extension IN ('.py', '.r') THEN 
            CASE 
                WHEN file_content LIKE '%import %' OR file_content LIKE '%from ... import%' THEN 
                    'Python'
                WHEN file_content LIKE '%library(%' THEN 
                    'R'
                ELSE 
                    NULL
            END
        ELSE 
            NULL
    END AS language,
    COUNT(*) AS occurrence_count
FROM (
    SELECT 
        repo_id,
        file_path,
        SUBSTRING_INDEX(file_path, '.', -1) AS file_extension,
        file_content
    FROM 
        github_repos_files
    JOIN 
        github_repos_contents ON github_repos_files.repo_id = github_repos_contents.repo_id AND github_repos_files.file_id = github_repos_contents.file_id
) AS joined_data
WHERE 
    file_extension IN ('.py', '.r')
GROUP BY 
    language, file_extension
ORDER BY 
    language ASC, occurrence_count DESC;
```