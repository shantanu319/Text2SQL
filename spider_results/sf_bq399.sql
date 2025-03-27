```sql
SELECT 
    r.region_name,
    h.country_name,
    AVG(c.birth_rate) AS avg_crude_birth_rate
FROM 
    regions r
JOIN 
    countries c ON r.region_id = c.region_id
JOIN 
    high_income_countries h ON c.country_id = h.country_id
WHERE 
    c.year BETWEEN 1980 AND 1989
GROUP BY 
    r.region_name, h.country_name
ORDER BY 
    avg_crude_birth_rate DESC;
```