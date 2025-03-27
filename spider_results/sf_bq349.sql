```sql
SELECT p.id AS osm_id
FROM planet_features p
JOIN planet_nodes n ON ST_Intersects(p.geometry, n.geometry)
WHERE n.tags->>'amenity' IS NOT NULL
GROUP BY p.id
HAVING COUNT(n.tags->>'amenity') = (
    SELECT AVG(count)
    FROM (
        SELECT COUNT(*) AS count
        FROM planet_features pf
        JOIN planet_nodes pn ON ST_Intersects(pf.geometry, pn.geometry)
        WHERE pn.tags->>'amenity' IS NOT NULL
        GROUP BY pf.id
    ) subquery
);
```