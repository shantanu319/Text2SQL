```sql
SELECT 
    pt.sample_type,
    AVG(CORR(protein_abundance_log2_ratio, gene_expression_log_transformed_plus_one_FPKM)) AS avg_correlation
FROM 
    (
        SELECT 
            s.sample_id,
            s.sample_type,
            p.protein_abundance_log2_ratio,
            r.gene_expression_log_transformed_plus_one_FPKM
        FROM 
            CPTAC_proteomics p
        JOIN 
            CPTAC_rnaseq r ON p.sample_submitter_id = r.sample_submitter_id
        JOIN 
            samples s ON p.sample_id = s.sample_id
        WHERE 
            s.sample_type IN ('Primary Tumor', 'Solid Tissue Normal')
    ) AS joined_data
GROUP BY 
    pt.sample_type;
```