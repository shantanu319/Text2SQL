```sql
SELECT 
    DATE(recorded_date) AS recorded_date,
    SUM(CASE WHEN status IN ('90-180 Days Past Due', 'Forbearance', 'Foreclosure', 'Bankruptcy', 'Deed-In-Lieu') THEN 1 ELSE 0 END) / COUNT(*) AS cumulative_ratio
FROM 
    mortgage_data
WHERE 
    state = 'California' AND YEAR(recorded_date) = 2023
GROUP BY 
    DATE(recorded_date)
ORDER BY 
    recorded_date;
```

This SQL query calculates the cumulative ratio of mortgages near default in California for each recorded date in 2023. It includes mortgages that are 90 to 180 days past due, in forbearance, or undergoing foreclosure, bankruptcy, or deed-in-lieu processes. The results are grouped by recorded date and ordered chronologically.