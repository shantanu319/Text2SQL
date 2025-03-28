```sql
SELECT 
    T3.game_id,
    T3.team_market,
    T3.player_name,
    T3.score,
    T3.period,
    T3.year
FROM 
    (
        SELECT 
            DISTINCT player_name, 
            score, 
            period, 
            year
        FROM 
            ncaa_basketball_historical_tournament_matches
        WHERE 
            year BETWEEN 2010 AND 2018
            AND period = 2
            AND score >= 15
        GROUP BY 
            player_name, 
            score, 
            period, 
            year
        ORDER BY 
            COUNT(DISTINCT player_name) DESC
        LIMIT 5
    ) AS T1
JOIN 
    ncaa_basketball_historical_tournament_matches AS T2 ON T1.player_name = T2.player_name
JOIN 
    ncaa_basketball_historical_tournament_matches AS T3 ON T2.game_id = T3.game_id
WHERE 
    T3.year BETWEEN 2010 AND 2018;
```

This SQL query retrieves the details of each game played by the top five team markets with the highest number of distinct players who scored at