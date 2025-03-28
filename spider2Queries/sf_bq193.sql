To achieve the desired result, you can use the following SQL query. This query assumes you have access to a database that contains information about GitHub repositories and their README.md files.

```sql
WITH LineCounts AS (
    SELECT 
        repo_id,
        line_number,
        COUNT(*) AS line_count
    FROM 
        github_readme_lines
    WHERE 
        line_content NOT LIKE '#%'
        AND line_content NOT LIKE '//%'
        AND line_content <> ''
    GROUP BY 
        repo_id, 
        line_number
),
LanguageCounts AS (
    SELECT 
        repo_id,
        language_name
    FROM 
        github_repo_languages
)
SELECT 
    lc.line_number,
    lc.line_count,
    STRING_AGG(lc.language_name, ', ') WITHIN GROUP (ORDER BY lc.language_name) AS languages_used
FROM 
    LineCounts lc
JOIN 
    LanguageCounts lc2 ON lc.repo_id = lc2.repo_id
GROUP BY 
    lc.line_number, 
    lc.line_count
ORDER BY 
    lc.line_count DESC;
```

### Explanation:

1. **LineCounts CTE**: This Common Table Expression (CTE) calculates the number of occurrences of each non-empty, non-commented line in the README.md files