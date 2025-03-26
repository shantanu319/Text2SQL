SELECT COUNT(*) FROM (
    SELECT DISTINCT country_name 
    FROM world_bank_global_population_dataset, world_bank_health_nutrition_population_dataset
    WHERE world_bank_global_population_dataset.year = '2018' AND world_bank_health_nutrition_population_dataset.year = '2017'
        AND world_bank_global_population_dataset.total_population > world_bank_health_nutrition_population_dataset.total_population * 1.01
        AND world_bank_global_population_dataset.per_capita_current_health_expenditure_ppp > world_bank_health_nutrition_population_dataset.per_capita_current_health_expenditure_ppp * 1.01