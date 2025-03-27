```sql
SELECT 
    T2.library_name,
    COUNT(*) AS import_count
FROM 
    github_sample_dataset AS T1
JOIN 
    (
        SELECT 
            file_path,
            SUBSTRING_INDEX(SUBSTRING_INDEX(content, 'import ', -1), ',', 1) AS library_name
        FROM 
            github_sample_dataset
        WHERE 
            content LIKE '%.py' OR content LIKE '%.r' OR content LIKE '%.R' OR content LIKE '%.Rmd' OR content LIKE '%.rmd' OR content LIKE '%.ipynb'
    ) AS T2 ON T1.file_path = T2.file_path
GROUP BY 
    T2.library_name
ORDER BY 
    import_count DESC
LIMIT 1 OFFSET 1;
```

This SQL query joins the `github_sample_dataset` table with a subquery that extracts the first library name from each import statement in Python, R, R Markdown, and IPython Notebook files. It then groups by the library name and counts the occurrences of each library. The results are ordered by the count in descending order, and the second most frequently imported library is selected using `LIMIT 1 OFFSET 1`.