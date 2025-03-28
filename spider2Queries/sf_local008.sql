```sql
SELECT DISTINCT T1.given_name
FROM baseball_players AS T1
JOIN (
    SELECT player_id, MAX(T2.games_played + T2.runs + T2.hits + T2.home_runs) AS max_score
    FROM baseball_stats AS T2
    GROUP BY player_id
) AS T3 ON T1.player_id = T3.player_id;
```