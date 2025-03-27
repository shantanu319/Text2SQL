```sql
SELECT 
    T1.variant_density,
    MAX(T2.reference_length) AS max_reference_length
FROM 
    cannabis_references AS T1
JOIN 
    variants AS T2 ON T1.cannabis_reference_id = T2.cannabis_reference_id
WHERE 
    T2.genotype > 0
GROUP BY 
    T1.variant_density;
```