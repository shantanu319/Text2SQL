SELECT census_tract FROM census_data WHERE (population_increase_percentage > 20 AND median_income_absolute_increase > 20) AND num_residents_2011 >= 1000 AND num_residents_2018 >= 1000