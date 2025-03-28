```sql
SELECT T1.title
FROM questions AS T1
JOIN tags AS T2 ON T1.id = T2.question_id
WHERE T2.tag_name IN ('android-layout', 'android-activity', 'android-intent')
  AND T1.tags LIKE '%android%'
ORDER BY T1.view_count DESC
LIMIT 1;
```

This query selects the title of the most viewed "how" question related to Android development from Stack Overflow, considering only questions tagged with 'android-layout', 'android-activity', 'android-intent', and other relevant tags. It orders the results by view count in descending order and limits the output to the top result. Note that the `tags` column in the `questions` table contains a comma-separated list of tag names, so we use the `LIKE` operator to filter questions based on these tags. Additionally, we ensure that the question is tagged with 'android' using the condition `T1.tags LIKE '%android%'`.