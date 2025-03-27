```sql
SELECT 
    s.state_name,
    COUNT(wt.employee_id) AS vulnerable_wholesale_trade_workers,
    COUNT(mf.employee_id) AS vulnerable_manufacturing_workers,
    (COUNT(wt.employee_id) + COUNT(mf.employee_id)) AS total_vulnerable_workers
FROM 
    states s
JOIN 
    zip_level_median_income_differences d ON s.zip_code = d.zip_code
JOIN 
    acs_employment_figures e ON s.zip_code = e.zip_code
LEFT JOIN 
    wholesale_trade_employees wt ON e.employee_id = wt.employee_id AND e.year = 2017
LEFT JOIN 
    manufacturing_employees mf ON e.employee_id = mf.employee_id AND e.year = 2017
WHERE 
    d.year = 2018 - 2015
GROUP BY 
    s.state_name
ORDER BY 
    total_vulnerable_workers DESC;
```

This SQL query joins multiple tables to calculate the number of vulnerable workers for each state based on the given criteria. It uses `LEFT JOIN` to ensure that all states are included in the result, even if they do not have data for certain years or employee types. The results are then grouped by state and ordered