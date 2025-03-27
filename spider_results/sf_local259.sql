```sql
SELECT 
    p.id,
    p.name,
    (
        SELECT r.role
        FROM roles r
        JOIN player_roles pr ON r.id = pr.role_id
        WHERE pr.player_id = p.id
        GROUP BY r.role
        ORDER BY COUNT(*) DESC
        LIMIT 1
    ) AS most_frequent_role,
    p.batting_hand,
    p.bowling_skill,
    SUM(p.runs) AS total_runs_scored,
    COUNT(m.match_id) AS total_matches_played,
    SUM(CASE WHEN m.dismissal_type IN ('out', 'lbw') THEN 1 ELSE 0 END) AS total_times_dismissed,
    CAST(SUM(p.runs) AS REAL) / COUNT(CASE WHEN m.dismissal_type IN ('out', 'lbw') THEN 1 ELSE 0 END) AS batting_average,
    MAX(p.highest_score) AS highest_score_in_single_match,
    COUNT(CASE WHEN p.runs >= 30 THEN 1 ELSE NULL END) AS matches_30_runs,
    COUNT(CASE WHEN p.runs >= 50 THEN 1 ELSE NULL END) AS matches_50_runs,
    COUNT(CASE WHEN p.runs