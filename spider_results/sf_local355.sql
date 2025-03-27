```sql
SELECT 
    YEAR(race_date) AS race_year,
    AVG(first_round_missed) AS avg_first_round_missed,
    AVG(last_round_missed) AS avg_last_round_missed
FROM 
    races
WHERE 
    driver_id IN (
        SELECT 
            driver_id
        FROM 
            drivers
        WHERE 
            COUNT(CASE WHEN race_missed = 1 THEN 1 END) < 3
    )
    AND race_missed = 1
GROUP BY 
    YEAR(race_date)
HAVING 
    (race_date - LAG(race_date, 1) OVER (PARTITION BY driver_id ORDER BY race_date)) <= 1
    OR (LEAD(race_date, 1) OVER (PARTITION BY driver_id ORDER BY race_date) - race_date) <= 1;
```

This SQL query calculates the average number of first and last rounds of races missed by drivers each year, considering only those drivers who missed fewer than three races annually and who switched teams between the race immediately before their first missed race and the race immediately after their last missed race. The query uses window functions to identify consecutive races for each driver and filters based on these conditions. It then groups the results