```sql
SELECT 
    q.Id AS QuestionId,
    q.Title AS QuestionTitle,
    q.Body AS QuestionBody,
    u.Reputation AS UserReputation,
    u.UpVotes - u.DownVotes AS NetVotes,
    COUNT(b.BadgeName) AS TotalBadges
FROM 
    Posts q
JOIN 
    Users u ON q.OwnerUserId = u.Id
LEFT JOIN 
    PostHistory ph ON q.Id = ph.PostId AND ph.PostHistoryTypeId = 2 -- Accepted Answer
LEFT JOIN 
    Votes v ON q.Id = v.PostId AND v.VoteTypeId = 2 -- Up Vote
LEFT JOIN 
    Badges b ON u.Id = b.UserId
WHERE 
    q.PostTypeId = 1 -- Questions
    AND (ph.AcceptedAnswerPostId IS NOT NULL OR (
        ph.AcceptedAnswerPostId IS NULL
        AND EXISTS (
            SELECT 1
            FROM Votes vs
            WHERE vs.PostId = q.Id
              AND vs.VoteTypeId = 2
              AND vs.Score > 0.01 * q.ViewCount
        )
    ))
GROUP BY 
    q.Id, q.Title, q.Body, u.Reputation, u.UpVotes, u.DownVotes