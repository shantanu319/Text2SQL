```sql
SELECT 
    bowler_name,
    COUNT(CASE WHEN dismissal_type NOT IN ('run-out', 'other') THEN 1 END) AS total_wickets,
    SUM(runs_conceded - (CASE WHEN is_extra = TRUE THEN runs_conceded ELSE 0 END)) / total_overs AS economy_rate,
    total_wickets / AVG(total_overs) AS strike_rate,
    MAX(CONCAT(wickets_taken, '-', runs_conceded - (CASE WHEN is_extra = TRUE THEN runs_conceded ELSE 0 END))) AS best_performance
FROM 
    bowlers
JOIN 
    matches ON bowlers.match_id = matches.id
JOIN 
    deliveries ON bowlers.delivery_id = deliveries.id
WHERE 
    dismissal_type NOT IN ('run-out', 'other')
GROUP BY 
    bowler_name;
```

This SQL query calculates the required statistics for each bowler based on the given conditions. It joins the `bowlers`, `matches`, and `deliveries` tables to gather the necessary data. The `COUNT` function is used to count the valid wickets, while the `SUM` function calculates the total runs conceded, excluding extras. The `AVG` function computes the average overs bowled per w