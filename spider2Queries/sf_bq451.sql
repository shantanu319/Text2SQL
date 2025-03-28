```sql
SELECT 
    sample_id,
    SUM(CASE WHEN genotype_array = '0' THEN 1 ELSE 0 END) AS homozygous_reference,
    SUM(CASE WHEN genotype_array = '1' THEN 1 ELSE 0 END) AS homozygous_alternate,
    SUM(CASE WHEN genotype_array != '0' AND genotype_array != '1' THEN 1 ELSE 0 END) AS heterozygous,
    COUNT(*) AS total_callable_sites,
    SUM(CASE WHEN genotype_array = '1' THEN 1 ELSE 0 END) AS total_heterozygous_snvs,
    SUM(CASE WHEN genotype_array = '1' THEN 1 ELSE 0 END) / COUNT(*) * 100 AS percentage_heterozygous_snvs,
    SUM(CASE WHEN genotype_array = '1' THEN 1 ELSE 0 END) / SUM(CASE WHEN genotype_array != '0' THEN 1 ELSE 0 END) * 100 AS percentage_homozygous_alternate_snvs
FROM 
    genotype_data
WHERE 
    chromosome = 'X'
    AND start NOT BETWEEN 59999 AND 269951