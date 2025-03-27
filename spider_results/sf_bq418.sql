```sql
SELECT 
    rp.pathway_id,
    rp.pathway_name,
    COUNT(DISTINCT CASE WHEN t.target_type = 'Homo sapiens' THEN t.target_id ELSE NULL END) AS target_count,
    COUNT(DISTINCT CASE WHEN t.target_type != 'Homo sapiens' THEN t.target_id ELSE NULL END) AS non_target_count
FROM 
    reactome_pathways rp
JOIN 
    reactome_pathway_targets rpt ON rp.pathway_id = rpt.pathway_id
JOIN 
    targets t ON rpt.target_id = t.target_id
WHERE 
    rp.evidence_type = 'TAS'
    AND rp.taxonomy_id = 9606
    AND rp.chi_squared_statistic > (
        SELECT 
            MAX(chi_squared_statistic)
        FROM 
            reactome_pathways
        WHERE 
            taxonomy_id = 9606
            AND evidence_type = 'TAS'
            AND pathway_level < rp.pathway_level
    )
    AND rp.median_assay_value <= 100
    AND rp.lowest_assay_value <= 100
    OR rp.lowest_assay_value IS NULL
    AND rp.highest_assay_value <=