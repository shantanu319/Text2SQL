```sql
SELECT 
    T3.common_name,
    COUNT(*) AS total_trees,
    SUM(CASE WHEN T4.year = 1995 THEN 1 ELSE 0 END) AS alive_1995,
    SUM(CASE WHEN T4.year = 1995 THEN 0 ELSE 1 END) AS dead_1995,
    SUM(CASE WHEN T4.year = 2015 THEN 1 ELSE 0 END) AS alive_2015,
    SUM(CASE WHEN T4.year = 2015 THEN 0 ELSE 1 END) AS dead_2015,
    (SUM(CASE WHEN T4.year = 2015 THEN 1 ELSE 0 END) - SUM(CASE WHEN T4.year = 1995 THEN 1 ELSE 0 END)) AS growth_difference
FROM 
    tree_species AS T1
JOIN 
    latin_names AS T2 ON T1.latin_name_id = T2.id
LEFT JOIN 
    common_names AS T3 ON T1.common_name_id = T3.id
LEFT JOIN 
    tree_counts AS T4 ON T1.tree_count_id = T4.id