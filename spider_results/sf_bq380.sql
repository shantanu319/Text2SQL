```sql
SELECT 
    u.Username,
    COUNT(DISTINCT mv.ToUserId) AS TotalUpVotesReceived,
    COUNT(DISTINCT mv.FromUserId) AS TotalUpVotesGiven
FROM 
    ForumMessageVotes mv
JOIN 
    Users u ON mv.FromUserId = u.UserId
GROUP BY 
    u.Username
ORDER BY 
    TotalUpVotesReceived DESC
LIMIT 3;
```