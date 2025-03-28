```sql
SELECT SUM(CASE WHEN T2.property_damage > 0 THEN 1 ELSE 0 END) AS total_severe_storm_events
FROM (
    SELECT 
        DATE_FORMAT(T1.event_date, '%Y-%m') AS event_month,
        COUNT(*) AS num_events
    FROM 
        severe_storms AS T1
    WHERE 
        T1.event_date >= DATE_SUB(NOW(), INTERVAL 15 YEAR)
    GROUP BY 
        event_month
    ORDER BY 
        num_events DESC
    LIMIT 1
) AS top_months
JOIN 
    severe_storms AS T2 ON DATE_FORMAT(T2.event_date, '%Y-%m') = top_months.event_month
WHERE 
    T2.property_damage > 0;
```