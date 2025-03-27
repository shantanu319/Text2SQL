```sql
SELECT 
    s.state_abbr,
    SUM(CASE WHEN t.year = 2015 THEN t.median_income_change ELSE NULL END) AS median_income_change_2015,
    SUM(CASE WHEN t.year = 2016 THEN t.median_income_change ELSE NULL END) AS median_income_change_2016,
    SUM(CASE WHEN t.year = 2017 THEN t.median_income_change ELSE NULL END) AS median_income_change_2017,
    SUM(CASE WHEN t.year = 2018 THEN t.median_income_change ELSE NULL END) AS median_income_change_2018,
    SUM(t.adjusted_non_labor_force_count_2017) AS adjusted_non_labor_force_count_2017,
    SUM(t.total_population) AS total_population,
    SUM(t.adjusted_non_labor_force_count_2017) / SUM(t.total_population) AS population_adjusted_proportion
FROM 
    states s
JOIN 
    tracts t ON s.fips_code_prefix = LEFT(t.fips_code, LENGTH(s.fips_code_prefix))
WHERE 
    t.income_difference IS NOT NULL
GROUP BY 
    s.state