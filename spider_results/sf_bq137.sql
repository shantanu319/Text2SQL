```sql
SELECT 
    zca.zip_code_area_polygon,
    zca.land_area_meters,
    zca.water_area_meters,
    zca.latitude,
    zca.longitude,
    zca.state_code,
    zca.state_name,
    zca.city,
    zca.county,
    SUM(census_data.male_population + census_data.female_population) AS total_population
FROM 
    zip_code_areas zca
JOIN 
    census_data ON zca.zip_code = census_data.zip_code
WHERE 
    ST_Distance(zca.geometry, ST_GeomFromText('POINT(-122.3321 47.6062)', 4326)) <= 10000
GROUP BY 
    zca.zip_code_area_polygon,
    zca.land_area_meters,
    zca.water_area_meters,
    zca.latitude,
    zca.longitude,
    zca.state_code,
    zca.state_name,
    zca.city,
    zca.county;
```

This SQL query joins the `zip_code_areas` table with the `census_data` table based on the zip code. It then filters the results to include only those zip code areas that