```sql
SELECT g.gene_id, v.variant_id, v.p_value
FROM (
    SELECT gene_id, MIN(p_value) AS min_p_value
    FROM gene_study_variant
    GROUP BY gene_id
) g
JOIN gene_study_variant v ON g.gene_id = v.gene_id AND g.min_p_value = v.p_value
ORDER BY v.p_value ASC
LIMIT 10;
```