```sql
SELECT 
    state_name,
    SUM(weighted_sum) AS total_vulnerable_population,
    AVG(median_income_change) AS average_median_income_change
FROM (
    SELECT 
        state_name,
        SUM(employment_sector_weight * employment_sector_value) AS weighted_sum,
        z.zip_code,
        (z.median_income_2018 - z.median_income_2015) / 3 AS median_income_change
    FROM 
        us_states s
    JOIN 
        employment_sectors es ON s.state_id = es.state_id
    JOIN 
        zip_codes z ON s.state_id = z.state_id
    GROUP BY 
        state_name, z.zip_code
) AS subquery
GROUP BY 
    state_name
ORDER BY 
    total_vulnerable_population DESC
LIMIT 10;
```

This SQL query calculates the top 10 U.S. states with the highest vulnerable population based on a weighted sum of employment sectors using 2017 ACS 5-Year data. It also determines their average median income change from 2015 to 2018 using zip code data. The results are ordered by the total vulnerable population in descending order and limited to the top 10