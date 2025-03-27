```sql
SELECT 
    coc_region,
    (SUM(homeless_veterans_2018) - SUM(homeless_veterans_2012)) AS change_in_homelessness
FROM 
    new_york_data
WHERE 
    year IN (2012, 2018)
GROUP BY 
    coc_region;
```