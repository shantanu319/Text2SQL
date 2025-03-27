```sql
SELECT 
    package_name,
    COUNT(*) AS import_count
FROM (
    SELECT 
        REGEXP_EXTRACT(line_content, r'import\s+([a-zA-Z0-9.]+);') AS package_name
    FROM (
        SELECT 
            line_number,
            line_content
        FROM 
            github_repos_contents
        WHERE 
            file_extension = 'java'
    ) AS java_files
) AS filtered_imports
GROUP BY 
    package_name
ORDER BY 
    import_count DESC
LIMIT 10;
```

This SQL query performs the following steps:
1. It selects the `line_number` and `line_content` from the `github_repos_contents` table where the `file_extension` is 'java'.
2. It uses a regular expression (`REGEXP_EXTRACT`) to extract the package name from each line that starts with "import".
3. It groups the results by the extracted package name.
4. It counts the number of times each package appears.
5. It orders the results by the count in descending order.
6. Finally, it limits the output to the top 10 most frequently imported package names.