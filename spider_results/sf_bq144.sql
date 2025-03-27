```sql
CREATE TABLE ncaa_basketball_dataset AS
SELECT 
    t1.season,
    CASE WHEN t1.winner = 'Team A' THEN 1 ELSE 0 END AS win_label,
    t1.seed_a,
    t1.seed_b,
    t1.school_a,
    t1.school_b,
    t2.pace_ranking_a,
    t2.efficiency_ranking_a,
    t2.statistic_values_a,
    t2.difference_a,
    t3.pace_ranking_b,
    t3.efficiency_ranking_b,
    t3.statistic_values_b,
    t3.difference_b
FROM 
    ncaa_tournament_games_2014_onwards t1
LEFT JOIN 
    feature_engineering_data t2 ON t1.team_a_id = t2.team_id AND t1.game_date BETWEEN t2.start_date AND t2.end_date
LEFT JOIN 
    feature_engineering_data t3 ON t1.team_b_id = t3.team_id AND t1.game_date BETWEEN t3.start_date AND t3.end_date;
```

This SQL query creates a new table named `ncaa_basketball_dataset` that combines the necessary information from the `ncaa