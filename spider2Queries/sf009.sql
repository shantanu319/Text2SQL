```sql
SELECT 
    building_class,
    building_subclass,
    SUM(surface_area) AS total_surface_area,
    COUNT(*) AS number_of_buildings
FROM 
    buildings
WHERE 
    city IN ('Amsterdam', 'Rotterdam')
GROUP BY 
    building_class, 
    building_subclass;
```