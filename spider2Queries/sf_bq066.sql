```sql
SELECT 
    YEAR,
    AVG(poverty_rate) AS avg_poverty_rate,
    AVG(births_without_maternal_morbidity) AS avg_births_without_maternal_morbidity,
    CORR(avg_poverty_rate, avg_births_without_maternal_morbidity) AS pearson_correlation_coefficient
FROM (
    SELECT 
        YEAR,
        census_data.year AS census_year,
        census_data.poverty_rate,
        birth_data.births_without_maternal_morbidity
    FROM 
        census_data
    JOIN 
        birth_data ON census_data.year = birth_data.year - 5
    WHERE 
        birth_data.maternal_morbidity IS NULL
) AS subquery
GROUP BY 
    YEAR;
```