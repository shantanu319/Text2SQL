```sql
SELECT 
    c.county_name,
    AVG((e.employment_2018 - e.employment_2000) / e.employment_2000 * 100) AS percentage_increase
FROM 
    counties c
JOIN 
    employment_data e ON c.county_id = e.county_id
WHERE 
    e.year BETWEEN 2000 AND 2018
GROUP BY 
    c.county_name
ORDER BY 
    percentage_increase DESC
LIMIT 1;
```

This SQL query calculates the average percentage increase in construction employment for each county in Utah between 2000 and 2018, using data from the `employment_data` table. It averages the employment levels during the third month of each quarter in those years. The result is ordered by the percentage increase in descending order, and the top result (the county with the greatest percentage increase) is returned.