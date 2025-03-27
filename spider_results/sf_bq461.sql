```sql
SELECT 
    game_clock,
    wildcats_score + fighting_irish_score AS total_score,
    CASE 
        WHEN team = 'Wildcats' THEN wildcats_score
        ELSE fighting_irish_score
    END AS team_score,
    event_description
FROM 
    scoring_plays
WHERE 
    game_date BETWEEN '2014-01-01' AND '2014-12-31'
    AND home_team = 'Wildcats'
    AND away_team = 'Fighting Irish'
ORDER BY 
    game_clock;
```

This SQL query retrieves the required information from the `scoring_plays` table, filtering for games in the 2014 season where the Wildcats were the home team and the Fighting Irish were the away team. It calculates the total score for each play by summing the scores of both teams, determines which team scored based on the `team` column, and orders the results chronologically by the `game_clock`. The `event_description` is included as part of the output. Adjust the table and column names as necessary to fit your actual database schema.