```sql
SELECT username
FROM (
    SELECT username, COUNT(*) AS node_count
    FROM planet_nodes
    WHERE amenity IN ('hospital', 'clinic', 'doctors')
      AND ST_Within(geom, ST_MakeEnvelope(31.1798246, 18.4519921, 54.3798246, 33.6519921))
    GROUP BY username
) AS subquery
ORDER BY node_count DESC
LIMIT 3;
```