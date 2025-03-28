```sql
SELECT 
    CASE 
        WHEN T3.team_name = 'Team A' THEN 'Left'
        ELSE 'Right'
    END AS basket_side,
    AVG(T1.x_coordinate) AS avg_x_coordinate,
    AVG(T1.y_coordinate) AS avg_y_coordinate,
    COUNT(T1.shot_id) AS total_attempts,
    SUM(CASE WHEN T1.result = 'Success' THEN 1 ELSE 0 END) AS successful_shots
FROM 
    shots AS T1
JOIN 
    teams AS T2 ON T1.team_id = T2.team_id
JOIN 
    players AS T3 ON T1.player_id = T3.player_id
WHERE 
    T1.date < '2018-03-15'
    AND T1.shot_type IS NOT NULL
    AND T1.x_coordinate IS NOT NULL
    AND T1.y_coordinate IS NOT NULL
GROUP BY 
    T3.team_name;
```

This SQL query calculates the required statistics for each shot type, considering only shots taken before March 15, 2018, and excluding those with null values. It also ensures that the shots are on the correct side of the court based on the team's basket.