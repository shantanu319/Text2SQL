```sql
SELECT 
    sp.super_population_name,
    COUNT(DISTINCT v.variant_id) AS num_common_variants,
    COUNT(DISTINCT s.sample_id) AS num_samples_with_variant,
    sp.total_population_size,
    v.variant_type,
    COUNT(s.sample_id) OVER (PARTITION BY sp.super_population_name) AS total_sample_count
FROM 
    super_populations sp
JOIN 
    variants v ON sp.super_population_id = v.super_population_id
LEFT JOIN 
    sample_variants sv ON v.variant_id = sv.variant_id
LEFT JOIN 
    samples s ON sv.sample_id = s.sample_id
WHERE 
    v.allele_frequency >= 0.05
    AND v.chromosome NOT IN ('X', 'Y', 'MT')
GROUP BY 
    sp.super_population_name, v.variant_id, v.variant_type;
```

This SQL query calculates the number of common autosomal variants with an allele frequency of at least 0.05, grouped by the number of samples within each super population that possess the variant. It also includes the total population size for each super population, the variant types, and the sample counts in the analysis. The query excludes sex chromosomes (X, Y, MT) from the analysis