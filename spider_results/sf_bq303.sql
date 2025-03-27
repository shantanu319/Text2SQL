```sql
SELECT 
    u.id AS user_id,
    q.tags AS question_tags,
    a.tags AS answer_tags,
    c.tags AS comment_tags
FROM 
    Users u
JOIN 
    Questions q ON u.id = q.user_id AND q.creation_date BETWEEN '2019-07-01' AND '2019-12-31'
JOIN 
    Answers a ON u.id = a.user_id AND a.creation_date BETWEEN '2019-07-01' AND '2019-12-31'
LEFT JOIN 
    Comments c ON u.id = c.user_id AND c.creation_date BETWEEN '2019-07-01' AND '2019-12-31'
WHERE 
    u.id BETWEEN 16712208 AND 18712208;
```

This SQL query retrieves the user ID, tags of the relevant question, tags of the answer, and tags of the comment for each contribution made by users with IDs between 16712208 and 18712208 on Stack Overflow within the specified date range. It