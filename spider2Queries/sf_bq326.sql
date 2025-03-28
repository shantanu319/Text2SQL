```sql
SELECT COUNT(DISTINCT country_name) AS number_of_countries
FROM (
    SELECT 
        country_name,
        SUM(total_population) OVER (PARTITION BY country_name ORDER BY year) AS cumulative_total_population,
        SUM(per_capita_current_health_expenditure_ppp) OVER (PARTITION BY country_name ORDER BY year) AS cumulative_per_capita_health_expenditure
    FROM 
        world_bank_global_population_dataset
    JOIN 
        world_bank_health_nutrition_population_dataset ON world_bank_global_population_dataset.country_name = world_bank_health_nutrition_population_dataset.country_name
    WHERE 
        year BETWEEN 2017 AND 2018
) AS yearly_data
WHERE 
    (cumulative_total_population - LAG(cumulative_total_population) OVER (PARTITION BY country_name ORDER BY year)) / LAG(cumulative_total_population) OVER (PARTITION BY country_name ORDER BY year) > 0.01
    AND (cumulative_per_capita_health_expenditure - LAG(cumulative_per_capita_health_expenditure) OVER (PARTITION BY country_name ORDER BY year)) / LAG(cumulative_per_capita_health_expenditure) OVER (PARTITION BY