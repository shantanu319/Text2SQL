```sql
SELECT 
    ROUND(T2.t_score, 2) AS t_score
FROM 
    (
        SELECT 
            GROUP_CONCAT(DISTINCT T1.sample_id ORDER BY T1.sample_id SEPARATOR ', ') AS sample_ids,
            COUNT(*) AS num_samples
        FROM 
            LGG_data AS T1
        INNER JOIN 
            Mutation_data AS T2 ON T1.sample_id = T2.sample_id
        WHERE 
            T2.gene_name = 'DRG2'
            AND T2.mutation_status = 'TP53 mutated'
            AND T2.filter_status = 'PASS'
            AND T1.normalization_method = 'log10(normalized_count + 1)'
        GROUP BY 
            T1.sample_id
        HAVING 
            num_samples >= 10
            AND STDDEV(T1.log_expression) > 0
    ) AS T1
INNER JOIN 
    t_score_md AS T2 ON T1.sample_ids = T2.sample_ids;
```

This SQL query calculates the t-score for comparing the mean expression levels of the gene DRG2 between two groups (TP53 mutated vs. non-mutated) in the Lower Grade Glioma (LGG