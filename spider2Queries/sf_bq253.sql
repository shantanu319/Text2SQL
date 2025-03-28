```sql
SELECT r.name
FROM relations r
JOIN planet_features pf ON r.id = pf.relation_id
WHERE pf.wikidata_item = 'Q1095'
  AND r.name IS NOT NULL
  AND r.tags->>'wikidata' IS NULL
  AND EXISTS (
    SELECT 1
    FROM planet_features pf2
    WHERE pf2.relation_id = r.id
      AND pf2.tags->>'wikidata' IS NOT NULL
  )
ORDER BY COUNT(pf.id) DESC
LIMIT 1;
```