```sql
SELECT 
    evg.evaluation_group,
    evg.evaluation_type,
    evg.condition_status_code,
    evg.evaluation_description,
    evg.state_code,
    evg.macroplot_acres,
    evg.subplot_acres
FROM 
    evaluation_groups evg
JOIN 
    (
        SELECT 
            evaluation_group,
            MAX(subplot_acres) AS max_subplot_acres
        FROM 
            evaluation_groups
        WHERE 
            YEAR = 2012
        GROUP BY 
            evaluation_group
    ) max_subplot_acres ON evg.evaluation_group = max_subplot_acres.evaluation_group AND evg.subplot_acres = max_subplot_acres.max_subplot_acres
ORDER BY 
    evg.subplot_acres DESC
LIMIT 10;
```