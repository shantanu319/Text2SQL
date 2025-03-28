```sql
SELECT 
    c.county_name,
    s.state_name,
    c.median_age,
    c.total_population,
    (c.confirmed_cases / c.total_population) * 100 AS confirmed_cases_per_100k,
    (c.deaths / c.total_population) * 100 AS deaths_per_100k,
    c.case_fatality_rate_percentage
FROM 
    counties c
JOIN 
    states s ON c.state_id = s.state_id
WHERE 
    c.population > 50000
    AND c.covid_case_fatality_rate_date = '2020-08-27'
ORDER BY 
    c.case_fatality_rate_percentage DESC
LIMIT 3;
```