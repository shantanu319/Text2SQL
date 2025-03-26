SELECT u1.username AS 'User 1', u2.username AS 'User 2', COUNT(DISTINCT u1.upvote) AS 'Upvotes Received', SUM(u2.upvote) AS 'Upvotes Given Back'
FROM ForumMessageVotes fmv
JOIN Users u1 ON fmv.user_id = u1.id
JOIN Users u2 ON fmv.voter_id = u2.id
WHERE u1.upvote > u2.upvote