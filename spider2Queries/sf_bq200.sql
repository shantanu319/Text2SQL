```sql
SELECT 
    p.first_name,
    p.last_name,
    MAX(ps.pitch_speed) AS max_pitch_speed
FROM 
    pitchers p
JOIN 
    (
        SELECT 
            ps.pitcher_id,
            ps.game_id,
            ps.pitch_speed
        FROM 
            regular_season_pitches ps
        UNION ALL
        SELECT 
            ps.pitcher_id,
            ps.game_id,
            ps.pitch_speed
        FROM 
            post_season_pitches ps
    ) ps ON p.id = ps.pitcher_id
WHERE 
    ps.pitch_speed > 0
GROUP BY 
    p.id, p.first_name, p.last_name;
```

This SQL query retrieves the full names of pitchers who achieved the highest non-zero pitch speeds for each team. It joins the `pitchers` table with a subquery that combines data from both the regular season and post-season pitches tables. The subquery filters out any zero pitch speeds and groups the results by pitcher ID, first name, and last name. Finally, it selects the maximum pitch speed for each group. Note that this assumes there is a `regular_season_pitches` and `post_season_pitches` table with columns `pitcher_id`, `game_id`, and `pitch_speed