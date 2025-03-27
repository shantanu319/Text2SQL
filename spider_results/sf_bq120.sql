```sql
SELECT 
    region_name,
    COUNT(*) AS total_households,
    SUM(CASE WHEN income < 20000 THEN 1 ELSE 0 END) AS low_income_households,
    CAST(SUM(CASE WHEN income < 20000 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) AS low_income_ratio
FROM 
    snap_participants
JOIN 
    counties ON snap_participants.county_id = counties.id
WHERE 
    year = 2017 AND month = 1
GROUP BY 
    region_name
HAVING 
    COUNT(*) > 0
ORDER BY 
    total_households DESC
LIMIT 10;
```

This SQL query performs the following steps:
1. Joins the `snap_participants` table with the `counties` table on the county ID.
2. Filters the records to include only those from January 1, 2017.
3. Groups the results by region name.
4. Hides regions where the total SNAP participation is zero.
5. Calculates the total number of households and the number of households earning under $20,000 for each region.
6. Computes the ratio of low-income