```sql
SELECT 
    T1.gene_name,
    T1.expression_level,
    T2.clinical_info,
    T3.somatic_mutations
FROM 
    rna_expression_levels AS T1
JOIN 
    clinical_data AS T2 ON T1.patient_id = T2.patient_id
JOIN 
    somatic_mutations AS T3 ON T1.patient_id = T3.patient_id
WHERE 
    T1.project = 'TCGA-BLCA'
    AND T1.gene_name IN ('MDM2', 'TP53', 'CDKN1A', 'CCNE1')
    AND T3.mutation_type = 'CDKN2A'
    AND T3.reference_genome = 'hg19'
    AND T3.date_of_reference = 'Feb 2017';
```