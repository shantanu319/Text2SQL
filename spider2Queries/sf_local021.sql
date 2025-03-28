```sql
SELECT AVG(total_runs) AS average_total_runs
FROM (
    SELECT SUM(runs_scored) AS total_runs
    FROM matches
    WHERE striker_id IN (
        SELECT striker_id
        FROM matches
        GROUP BY striker_id
        HAVING SUM(runs_scored) > 50
    )
    GROUP BY striker_id
) AS striker_totals;
```