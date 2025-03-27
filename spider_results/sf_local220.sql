```sql
SELECT 
    player_name,
    COUNT(*) AS total_matches,
    SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN result = 'lose' THEN 1 ELSE 0 END) AS losses
FROM 
    matches
WHERE 
    player_id IS NOT NULL AND 
    result IS NOT NULL
GROUP BY 
    player_name
ORDER BY 
    wins DESC, 
    losses DESC
LIMIT 2;
```

This query calculates the total number of matches each player has participated in, as well as the number of wins and losses. It then orders the results by the number of wins in descending order and by the number of losses in descending order, limiting the output to the top two players with the most wins and losses. The `result` column is assumed to be a string indicating whether the match was a win ('win') or a loss ('lose'). Adjust the table and column names as necessary to fit your database schema.