```sql
SELECT 
    COUNT(CASE WHEN T2.path LIKE '%README.MD%' THEN 1 ELSE NULL END) / COUNT(T1.id) AS proportion
FROM 
    repository AS T1
LEFT JOIN 
    file AS T2 ON T1.id = T2.repository_id
WHERE 
    NOT EXISTS (
        SELECT 1 FROM programming_language AS PL WHERE LOWER(PL.name) LIKE '%python%'
    )
    AND T2.contents LIKE '%Copyright (c)%';
```

This SQL query calculates the proportion of files in repositories that do not use Python as a programming language, have paths containing "README.md", and contain the phrase "Copyright (c)". It uses a `LEFT JOIN` to combine the `repository` and `file` tables based on the repository ID. The `NOT EXISTS` clause filters out repositories that use Python. The `COUNT` function with a `CASE` statement counts the number of files meeting the criteria, and the total count of files from each repository is used to calculate the proportion.