```sql
SELECT 
    T3.plot_type,
    T4.year,
    T4.state,
    AVG(T3.avg_subplot_size) AS avg_subplot_size,
    AVG(T3.avg_macroplot_size) AS avg_macroplot_size
FROM 
    `bigquery-public-data.usfs_fia.condition` AS T1
JOIN 
    `bigquery-public-data.usfs_fia.plot_tree` AS T2 ON T1.fid = T2.fid
JOIN 
    `bigquery-public-data.usfs_fia.plot` AS T3 ON T2.plot_id = T3.plot_id
JOIN 
    `bigquery-public-data.usfs_fia.evaluation` AS T4 ON T3.plot_id = T4.plot_id
WHERE 
    T1.evaluation_type = 'EXPCURR'
    AND T1.condition_status_code = 1
    AND T4.year IN (2015, 2016, 2017)
GROUP BY 
    T3.plot_type, T4.year, T4.state
ORDER BY 
    T3.plot_type, T4.year, T4.state;
```