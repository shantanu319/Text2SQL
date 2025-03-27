```sql
SELECT 
    u1.username AS user1,
    u2.username AS user2,
    COUNT(DISTINCT fmv.upvote_id) AS total_upvotes_received,
    SUM(CASE WHEN fmv.user_id = u1.id THEN 1 ELSE 0 END) AS upvotes_given_back
FROM 
    ForumMessageVotes fmv
JOIN 
    Users u1 ON fmv.user_id = u1.id
JOIN 
    Users u2 ON fmv.voter_id = u2.id
GROUP BY 
    u1.username, u2.username
ORDER BY 
    total_upvotes_received DESC, 
    SUM(CASE WHEN fmv.user_id = u1.id THEN 1 ELSE 0 END) DESC
LIMIT 1;
```