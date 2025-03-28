```sql
SELECT 
    u.username,
    m.message_score - AVG(m2.message_score) AS score_difference
FROM 
    forum_topics ft
JOIN 
    messages m ON ft.topic_id = m.topic_id AND m.is_first_message = 1
JOIN 
    users u ON m.author_id = u.user_id
LEFT JOIN 
    votes v ON m.message_id = v.message_id
LEFT JOIN (
    SELECT 
        author_id,
        COUNT(DISTINCT voter_id) AS message_score
    FROM 
        messages
    WHERE 
        is_first_message = 1
    GROUP BY 
        author_id
) m2 ON m.author_id = m2.author_id
GROUP BY 
    u.username, m.message_score
ORDER BY 
    m.message_score DESC
LIMIT 3;
```

This query joins the `forum_topics`, `messages`, and `users` tables to find the first messages in forum topics. It calculates the message score for each user by counting the distinct voters. The results are then grouped by user and ordered by message score in descending order. Finally, it limits the output to the top three users with the highest message scores. The absolute difference between each user's message score and the average message score across all