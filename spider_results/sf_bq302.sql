```sql
SELECT 
    DATE_TRUNC('month', created_at) AS month,
    COUNT(*) / COUNT(DISTINCT question_id) * 100 AS python_tag_proportion
FROM 
    stackoverflow_questions
WHERE 
    tags LIKE '%python%'
    AND EXTRACT(YEAR FROM created_at) = 2022
GROUP BY 
    DATE_TRUNC('month', created_at)
ORDER BY 
    month;
```