To convert the given text into an SQL query, we need to identify the relevant tables and columns. Assuming we have two tables: `Users` and `ForumMessages`, where:

- `Users` table contains columns: `username`, `forum_message_id`
- `ForumMessages` table contains columns: `message_id`, `upvotes`, `year`

Here's how you can write the SQL query:

```sql
SELECT u.username
FROM Users u
JOIN ForumMessages fm ON u.forum_message_id = fm.message_id
WHERE fm.year = 2019
GROUP BY u.username
ORDER BY AVG(fm.upvotes) DESC, u.username ASC
LIMIT 1;
```

### Explanation:
1. **Join**: We join the `Users` table (`u`) with the `ForumMessages` table (`fm`) on the `forum_message_id`.
2. **Filter by Year**: We filter the records to include only those from the year 2019.
3. **Group By Username**: We group the results by `username` to calculate the average upvotes for each user.
4. **Order By**: We order the results first by the average upvotes in descending order (to get the highest average), and then by the