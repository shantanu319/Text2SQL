```sql
SELECT MAX(Answers) AS Highest_Answers
FROM (
    SELECT COUNT(*) AS Answers
    FROM posts
    WHERE Tags LIKE '%Python 2%'
      AND PostTypeId = 1 -- Questions
      AND NOT EXISTS (
          SELECT *
          FROM posts AS p2
          WHERE p2.PostId = posts.ParentId
            AND p2.Tags LIKE '%Python 3%'
      )
) AS Question_Stats;
```