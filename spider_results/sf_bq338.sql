```sql
SELECT 
    census_tract,
    population_increase_percentage,
    median_income_absolute_increase,
    resident_count_2011,
    resident_count_2018
FROM 
    census_data
WHERE 
    area_code = '36047'
    AND population_increase_percentage IN (
        SELECT 
            population_increase_percentage
        FROM 
            census_data
        WHERE 
            area_code = '36047'
        ORDER BY 
            population_increase_percentage DESC
        LIMIT 20
    )
    AND median_income_absolute_increase IN (
        SELECT 
            median_income_absolute_increase
        FROM 
            census_data
        WHERE 
            area_code = '36047'
        ORDER BY 
            median_income_absolute_increase DESC
        LIMIT 20
    )
    AND resident_count_2011 > 1000
    AND resident_count_2018 > 1000;
```