```sql
SELECT 
    country,
    SUM(confirmed_cases) AS total_confirmed_cases,
    (SUM(confirmed_cases) / SUM(population)) * 100000 AS cases_per_100k
FROM 
    covid_data
WHERE 
    country IN ('United States', 'France', 'China', 'Italy', 'Spain', 'Germany', 'Iran')
    AND date = 'April 20, 2020'
GROUP BY 
    country;
```

This SQL query calculates the total number of confirmed COVID-19 cases and the number of cases per 100,000 people for the specified countries as of April 20, 2020. It uses a `WHERE` clause to filter the records based on the country names and the date. The `SUM` function is used to calculate the total confirmed cases and the population for each country. The result is grouped by country using the `GROUP BY` clause. Note that the table name (`covid_data`) and column names (`confirmed_cases`, `population`) should be adjusted according to your actual database schema. Additionally, ensure that the date format in the `WHERE` clause matches the format used in your database