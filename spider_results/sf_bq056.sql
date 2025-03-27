```sql
SELECT COUNT(DISTINCT T1.osm_id, T2.osm_id) AS overlapping_roads_count
FROM planet_osm_roads AS T1
JOIN planet_osm_roads AS T2 ON ST_Intersects(T1.way, T2.way)
WHERE T1.highway IN ('motorway', 'trunk', 'primary', 'secondary', 'residential')
  AND T2.highway IN ('motorway', 'trunk', 'primary', 'secondary', 'residential')
  AND NOT EXISTS (SELECT 1 FROM planet_osm_nodes WHERE ST_Contains(T1.way, planet_osm_nodes.way))
  AND NOT EXISTS (SELECT 1 FROM planet_osm_nodes WHERE ST_Contains(T2.way, planet_osm_nodes.way))
  AND NOT EXISTS (SELECT 1 FROM planet_osm_tags WHERE T1.osm_id = planet_osm_tags.osm_id AND planet_osm_tags.k = 'bridge');
```

This SQL query calculates the number of unique pairs of roads that meet the specified criteria. It uses `ST_Intersects` to find overlapping segments between roads, checks for the presence of specific highway tags, ensures no shared nodes, and excludes any bridges. The result is the count of such