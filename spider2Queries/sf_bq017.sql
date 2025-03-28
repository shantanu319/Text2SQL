```sql
SELECT highway_type, SUM(length) AS total_length
FROM planet_osm_line
WHERE highway IN ('motorway', 'trunk', 'primary', 'secondary', 'tertiary')
  AND ST_Intersects(ST_GeomFromWKT('MULTIPOLYGON((...))'), way)
GROUP BY highway_type
ORDER BY total_length DESC
LIMIT 5;
```

Note: The `ST_GeomFromWKT` function is used to create a geometry from the WKT representation of the multipolygon boundary. You need to replace `(...)` with the actual coordinates of the boundary as provided by Wikidata ID 'Q35'. Additionally, you may need to adjust the `highway` filter based on the specific requirements and data available in your database.