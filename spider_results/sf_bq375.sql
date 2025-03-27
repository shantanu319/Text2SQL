```sql
SELECT 
    CASE 
        WHEN filename LIKE '%.py' THEN 'Python'
        WHEN filename LIKE '%.c' THEN 'C'
        WHEN filename LIKE '%.ipynb' THEN 'Jupyter Notebook'
        WHEN filename LIKE '%.java' THEN 'Java'
        WHEN filename LIKE '%.js' THEN 'JavaScript'
    END AS file_type,
    COUNT(*) AS file_count
FROM 
    github_codebase
WHERE 
    path LIKE '/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%