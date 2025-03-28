```sql
SELECT 
    COUNT(DISTINCT icd_o_3_histology_code) AS total_groups,
    COUNT(*) AS total_samples,
    AVG(kruskal_wallis_h_score) AS kruskal_wallis_h_score
FROM (
    SELECT 
        icd_o_3_histology_code,
        COUNT(normalized_count) AS sample_count,
        AVG(log10(normalized_count + 1)) AS igf2_expression_log10,
        KURTOSIS(log10(normalized_count + 1)) AS kurtosis,
        SUM(CASE WHEN normalized_count IS NOT NULL THEN 1 ELSE 0 END) AS valid_samples
    FROM 
        lgg_patients
    WHERE 
        normalized_count IS NOT NULL
        AND icd_o_3_histology_code NOT LIKE '%[]%'
        AND icd_o_3_histology_code IN (
            SELECT 
                icd_o_3_histology_code
            FROM 
                lgg_patients
            GROUP BY 
                icd_o_3_histology_code
            HAVING 
                COUNT(*) > 1
        )
    GROUP BY 
        icd_o_3_histology_code
) AS grouped_data;
```