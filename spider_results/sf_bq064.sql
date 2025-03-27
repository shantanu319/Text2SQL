```sql
SELECT 
    z.zip_code,
    ROUND(SUM(p.population * t.area / SUM(t.area)), 1) AS total_population,
    ROUND(AVG(i.income * p.population / SUM(p.population)), 1) AS average_individual_income
FROM 
    `bigquery-public-data.census_usa.tracts_2017` t
JOIN 
    `bigquery-public-data.us_zips.zips` z ON ST_INTERSECTS(ST_GEOGPOINT(t.longitude, t.latitude), z.geom)
JOIN 
    `bigquery-public-data.census_usa.population_2017` p ON t.geoid = p.geoid
JOIN 
    `bigquery-public-data.census_usa.income_2017` i ON t.geoid = i.geoid
WHERE 
    ST_Distance(ST_GEOGPOINT(47.685833, -122.191667), ST_GEOGPOINT(z.longitude, z.latitude)) <= 5000
GROUP BY 
    z.zip_code
ORDER BY 
    average_individual_income DESC;
```

This query performs the following steps:
1. Joins the census tracts