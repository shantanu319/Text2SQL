```sql
SELECT 
    Team,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY SeasonGoals DESC) AS MedianSeasonGoals
FROM 
    Teams
GROUP BY 
    Team;
```