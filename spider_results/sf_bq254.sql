```sql
SELECT 
    name,
    COUNT(*) AS point_count
FROM 
    multipolygons
WHERE 
    geog_area = (SELECT geog_area FROM multipolygons WHERE wikidata_id = 'Q191')
    AND NOT EXISTS (
        SELECT 1
        FROM tags
        WHERE multipolygon_id = multipolygons.id
        AND tag_name = 'wikidata'
    )
GROUP BY 
    name
ORDER BY 
    point_count DESC
LIMIT 2;
```