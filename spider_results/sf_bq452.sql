```sql
SELECT 
    v.start_position,
    v.end_position,
    c.chi_squared_score
FROM 
    variants v
JOIN 
    (
        SELECT 
            variant_id,
            SUM(CASE WHEN super_population = 'EAS' THEN allele_count ELSE 0 END) AS case_allele_count,
            SUM(CASE WHEN super_population != 'EAS' THEN allele_count ELSE 0 END) AS control_allele_count,
            CASE 
                WHEN case_allele_count < 5 OR control_allele_count < 5 THEN NULL
                ELSE CHI_SQUARED(case_allele_count, control_allele_count)
            END AS chi_squared_score
        FROM 
            variant_alleles va
        JOIN 
            super_populations sp ON va.super_population_id = sp.id
        GROUP BY 
            variant_id
    ) c ON v.id = c.variant_id
WHERE 
    c.chi_squared_score >= 29.71679;
```

Note: The `CHI_SQUARED` function is assumed to be a custom function or a placeholder for the actual chi-squared calculation logic. You may need to adjust this part based on your specific database system or implementation. Additionally, ensure that the