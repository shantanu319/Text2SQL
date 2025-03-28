```sql
SELECT player_name, AVG(runs_per_match) AS batting_average
FROM matches
WHERE season = 5
GROUP BY player_name
ORDER BY batting_average DESC
LIMIT 5;
```