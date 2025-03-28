```sql
SELECT 
    T2.jurisdiction_code,
    COUNT(*) AS patent_count
FROM 
    assignees AS T1
JOIN 
    patents AS T2 ON T1.assignee_id = T2.assignee_id
WHERE 
    T1.category = 'A61'
GROUP BY 
    T2.jurisdiction_code
ORDER BY 
    patent_count DESC
LIMIT 5;
```