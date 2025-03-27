```sql
WITH RecentPopulationGrid AS (
    SELECT 
        centroid,
        population
    FROM 
        population_grid
    WHERE 
        date < '2023-01-01'
),
ConvexHullRegion AS (
    SELECT 
        ST_ConvexHull(centroid) AS convex_hull
    FROM 
        RecentPopulationGrid
),
HospitalCentroids AS (
    SELECT 
        ST_Centroid(geometry) AS centroid
    FROM 
        osm_planet
    WHERE 
        layer_code IN (2110, 2120)
),
DistanceToNearestHospital AS (
    SELECT 
        p.grid_id,
        p.population,
        h.centroid,
        ST_Distance(p.centroid, h.centroid) AS distance_to_nearest_hospital
    FROM 
        RecentPopulationGrid p
    JOIN 
        HospitalCentroids h ON ST_Intersects(p.convex_hull, h.centroid)
)
SELECT 
    MAX(population) AS total_population_of_farthest_grid_cell
FROM 
    DistanceToNearestHospital;
```

This SQL query performs the following steps:

1. **Recent Population Grid**: Selects the most recent 1km population grid data before