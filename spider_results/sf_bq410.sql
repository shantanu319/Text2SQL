SELECT s.state_name, SUM(IFNULL(i.income_change_2015_2018, 0)) AS median_income_change,
       SUM(CASE WHEN nlf_count < 0 THEN 0 ELSE nlf_count END) AS adjusted_non_labor_force_count,
       SUM(nlf_count / p.population) AS population_adjusted_proportion
FROM states s
JOIN (
    SELECT t.state_fips_code, t.tract_id, i.income_change_2015_2018, n.nlf_count, p.population
    FROM tracts t
    JOIN non_labor_force n ON t.state_fips_code = n.state_fips_code AND t.tract_id = n.tract_id
    JOIN population p ON t.state_fips_code = p.state_fips_code AND t.tract_id = p.tract_id
    WHERE i.income_change_2015_2018 IS NOT NULL
) i ON s.state_fips_code = i.state_fips_code