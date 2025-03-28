```sql
SELECT team_name, COUNT(*) AS win_count
FROM matches
GROUP BY team_name
ORDER BY win_count DESC
LIMIT 10;
```