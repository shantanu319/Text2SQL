```sql
SELECT l.name AS league_name, t.team_name AS team_name
FROM leagues l
JOIN teams t ON l.id = t.league_id
JOIN (
    SELECT 
        t.id AS team_id,
        SUM(CASE WHEN g.home_goals > g.away_goals THEN 1 ELSE 0 END) AS total_home_wins,
        SUM(CASE WHEN g.home_goals < g.away_goals THEN 1 ELSE 0 END) AS total_away_wins,
        SUM(CASE WHEN g.home_goals = g.away_goals THEN 1 ELSE 0 END) AS total_draws
    FROM 
        games g
    JOIN 
        teams t ON g.team_id = t.id
    GROUP BY 
        t.id
) gw ON t.id = gw.team_id
GROUP BY 
    l.name, t.team_name
ORDER BY 
    SUM(gw.total_home_wins + gw.total_away_wins + gw.total_draws) ASC
LIMIT 1;
```

This query calculates the total number of match wins (home and away) for each team in each league. It then selects the team with the fewest total wins, ensuring that if there are ties, only one team is returned per league.