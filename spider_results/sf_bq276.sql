```sql
SELECT 
    p.port_name,
    s.state_name,
    DISTINCT ts.year AS storm_year,
    COUNT(DISTINCT ts.name) AS total_storms,
    DISTINCT ts.name AS storm_names,
    AVG(ts.category) AS avg_storm_category,
    AVG(ts.wind_speed) AS avg_wind_speed,
    ST_AsText(p.geometry) AS port_geometry,
    ST_AsText(ts.geometry) AS storm_geometry
FROM 
    ports p
JOIN 
    states s ON p.state_id = s.state_id
LEFT JOIN 
    tropical_storms ts ON p.port_id = ts.port_id AND ts.wind_speed >= 35 AND ts.category >= 'Minimal Tropical Storm'
WHERE 
    p.region_number = 6585
GROUP BY 
    p.port_name, s.state_name;
```

This SQL query retrieves information about ports in region number 6585 that are located within U.S. state boundaries and have been affected by named storms in the North Atlantic basin with specific criteria. It includes details such as the port's name, state name, distinct years of storms, total count of distinct storms, distinct storm names, average storm category, average wind speed, and the respective geometries for both the port and