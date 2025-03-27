```sql
SELECT 
    ts.fall_color,
    SUM(CASE WHEN t.year = 1995 AND t.dead = 'N' THEN 1 ELSE -1 END) AS total_change
FROM 
    tree_species ts
JOIN 
    trees t ON ts.scientific_name = UPPER(t.species)
WHERE 
    t.year BETWEEN 1995 AND 2015
GROUP BY 
    ts.fall_color;
```

This SQL query calculates the change in the number of living trees for each fall color in New York City from 1995 to 2015. It joins the `tree_species` table with the `trees` table on the uppercase form of the scientific names, filters the data for years between 1995 and 2015, and sums up the differences between the number of trees not marked as dead in 1995 and the number of trees alive in 2015 for each species. The results are grouped by fall color and summed to determine the total change in the number of trees for each fall color.