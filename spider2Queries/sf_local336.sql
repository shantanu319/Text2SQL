```sql
SELECT 
    CASE 
        WHEN lap_number <= 5 THEN 'First Five Laps'
        ELSE 'After Fifth Lap'
    END AS lap_range,
    CASE 
        WHEN reason = 'Retirement' THEN 'Retirements'
        WHEN reason = 'Pit Stop' THEN 'Pit Stops'
        WHEN reason = 'Start-Related Overtake' THEN 'Start-Related Overtakes'
        WHEN reason = 'Standard On-Track Pass' THEN 'Standard On-Track Passes'
        ELSE 'Other'
    END AS category,
    COUNT(*) AS number_of_overtakes
FROM 
    race_results
WHERE 
    lap_number <= 5
GROUP BY 
    lap_range, category;
```