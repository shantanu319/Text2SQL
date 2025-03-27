```sql
SELECT 
    T3.tag_name,
    COUNT(T1.id) AS mention_count
FROM 
    hn_comments AS T1
JOIN 
    hn_questions AS T2 ON T1.question_id = T2.id
JOIN 
    stackoverflow_tags AS T3 ON T2.tags LIKE CONCAT('%|', T3.tag_name, '|%')
WHERE 
    T1.created_at >= '2014-01-01'
GROUP BY 
    T3.tag_name
ORDER BY 
    mention_count DESC
LIMIT 10;
```

This SQL query identifies and ranks the top 10 tags from Stack Overflow questions that were referenced in Hacker News comments on or after 2014. It counts how many times each question was mentioned and splits the questions' tag strings by the '|' delimiter, grouping by tag. The results are ordered by the count of mentions in descending order and limited to the top 10 tags.