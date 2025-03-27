```sql
SELECT 
    COUNT(DISTINCT sample_id) AS total_samples,
    COUNT(DISTINCT mutation_type) AS number_of_mutation_types,
    AVG(mean_square_between_groups) AS mean_square_between_groups,
    AVG(mean_square_within_groups) AS mean_square_within_groups,
    AVG(f_statistic) AS f_statistic
FROM 
    genetic_variants
JOIN 
    mutation_data ON genetic_variants.sample_id = mutation_data.sample_id
WHERE 
    genetic_variants.genetic_variant IN (
        SELECT genetic_variant FROM genetic_variants WHERE gene_name = 'TP53'
    )
GROUP BY 
    genetic_variants.sample_id;
```