```sql
SELECT 
    mutation_type,
    gender,
    vital_status,
    days_to_death,
    mdm2_expression_level,
    tp53_expression_level,
    cdkn1a_expression_level,
    ccne1_expression_level
FROM 
    genomics_data_commons_release_39
WHERE 
    patient_id IN (
        SELECT 
            patient_id
        FROM 
            bladder_cancer_patients
        WHERE 
            cdkn2a_mutation = 'yes'
    )
```

This SQL query selects the specified columns from the `genomics_data_commons_release_39` table where the patient ID is associated with a bladder cancer patient who has a mutation in the CDKN2A gene. It also includes the gene expression levels for the four downstream genes (MDM2, TP53, CDKN1A, and CCNE1). The subquery filters the patient IDs based on the presence of the CDKN2A mutation.