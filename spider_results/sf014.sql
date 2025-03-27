```sql
SELECT 
    z.zip_code,
    SUM(c.commuters) AS total_commuters,
    s.state_benchmark,
    p.state_population
FROM 
    zip_codes z
JOIN 
    commuters c ON z.zip_code = c.zip_code
JOIN 
    states s ON z.state_id = s.state_id
JOIN 
    populations p ON z.state_id = p.state_id
WHERE 
    c.duration > 1 AND YEAR(c.year) = 2021
GROUP BY 
    z.zip_code, s.state_benchmark, p.state_population
ORDER BY 
    total_commuters DESC
LIMIT 1;
```

This SQL query retrieves the New York State ZIP code with the highest number of commuters traveling over one hour in 2021 ACS data. It joins multiple tables to get the required information such as the zip code, total commuters, state benchmark, and state population. The results are ordered by the total commuters in descending order and limited to the top result.