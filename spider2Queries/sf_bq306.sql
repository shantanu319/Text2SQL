```sql
SELECT 
    t.tag_name,
    SUM(CASE WHEN v.vote_type_id = 2 THEN 10 ELSE 0 END) + SUM(CASE WHEN v.vote_type_id = 1 THEN 15 ELSE 0 END) AS total_score
FROM 
    posts p
JOIN 
    votes v ON p.post_id = v.post_id
JOIN 
    post_tags pt ON p.post_id = pt.post_id
JOIN 
    tags t ON pt.tag_id = t.tag_id
WHERE 
    p.owner_user_id = 1908967 AND
    p.creation_date < '2018-06-07' AND
    v.vote_type_id IN (2, 1)
GROUP BY 
    t.tag_name
ORDER BY 
    total_score DESC
LIMIT 10;
```