```sql
SELECT 
    T3.zip_code,
    T3.state_abbreviation,
    (T4.population - T4.population_5_years_prior) / T4.population_5_years_prior * 100 AS growth_rate
FROM 
    census_zip_code_tabulation_areas AS T1
JOIN 
    census_estimates AS T2 ON T1.zip_code = T2.zip_code
JOIN 
    census_population AS T3 ON T1.zip_code = T3.zip_code
WHERE 
    T2.year BETWEEN 2015 AND 2020
    AND T2.population >= 25000
ORDER BY 
    growth_rate DESC
LIMIT 1 OFFSET 1;
```

This query calculates the annual population growth rate for each zip code between 2015 and 2020, filters out those with a population estimate of at least 25,000 over a 5-year period, orders the results by growth rate in descending order, and selects the second-highest growth rate. It then retrieves the corresponding zip code, state abbreviation, and growth rate. The `OFFSET 1` clause is used to skip the first result, which corresponds to the highest growth rate.