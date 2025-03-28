```sql
SELECT 
    T2.team_name AS champion_team,
    T3.league,
    T4.country,
    SUM(CASE WHEN T1.result = 'W' THEN 3 ELSE 0 END) + SUM(CASE WHEN T1.result = 'T' THEN 1 ELSE 0 END) AS total_points
FROM 
    matches AS T1
JOIN 
    teams AS T2 ON T1.winner_id = T2.team_id
JOIN 
    leagues AS T3 ON T1.league_id = T3.league_id
JOIN 
    countries AS T4 ON T1.country_id = T4.country_id
GROUP BY 
    T2.team_name, T3.league, T4.country;
```

This SQL query analyzes the soccer match dataset to determine the champion team for each season across all countries and leagues. It awards 3 points for every win, 1 point for every tie, and 0 points for every loss. The result is grouped by the champion's team name, league, country, and the total points accumulated for each season.