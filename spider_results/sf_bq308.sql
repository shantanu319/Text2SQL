```sql
SELECT 
    DATE_TRUNC('day', created_at) AS question_date,
    EXTRACT(DOW FROM created_at) AS day_of_week,
    COUNT(*) AS total_questions,
    SUM(CASE WHEN answer_at IS NOT NULL AND answer_at - created_at <= INTERVAL '1 hour' THEN 1 ELSE 0 END) AS answered_within_one_hour,
    (SUM(CASE WHEN answer_at IS NOT NULL AND answer_at - created_at <= INTERVAL '1 hour' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) AS percentage_answered_within_one_hour
FROM 
    stackoverflow_questions
WHERE 
    created_at BETWEEN '2021-01-01' AND '2021-12-31'
GROUP BY 
    DATE_TRUNC('day', created_at),
    EXTRACT(DOW FROM created_at)
ORDER BY 
    question_date;
```