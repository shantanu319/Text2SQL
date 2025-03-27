```sql
SELECT 
    state,
    AVG(median_income_2015 - median_income_2018) AS avg_difference,
    COUNT(*) AS num_vulnerable_employees
FROM 
    zip_code_data
WHERE 
    industry IN ('Wholesale Trade', 'Natural Resources and Construction', 'Arts and Entertainment', 'Information', 'Retail Trade')
GROUP BY 
    state
ORDER BY 
    avg_difference DESC
LIMIT 5;
```