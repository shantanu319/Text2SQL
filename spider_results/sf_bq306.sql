SELECT t.tag_name FROM (
    SELECT q.id AS question_id, COUNT(DISTINCT a.id) AS accepted_answers_count, SUM(CASE WHEN v.vote_type_id = 2 THEN 1 ELSE 0 END) AS upvotes_count
    FROM questions q
    JOIN answers a ON q.id = a.question_id
    JOIN votes v ON a.id = v.votable_id AND v.user_id = 1908967
    WHERE v.created_at < '2018-06-07'
    GROUP BY q.id
)