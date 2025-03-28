```sql
SELECT 
    university,
    COUNT(*) AS peak_performance_seasons
FROM 
    ncaa_basketball_seasons
WHERE 
    season BETWEEN 1900 AND 2000
    AND wins = (SELECT MAX(wins) FROM ncaa_basketball_seasons WHERE season BETWEEN 1900 AND 2000)
    AND university IS NOT NULL
GROUP BY 
    university
ORDER BY 
    peak_performance_seasons DESC
LIMIT 5;
```