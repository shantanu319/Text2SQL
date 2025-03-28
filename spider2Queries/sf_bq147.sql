```sql
SELECT *
FROM tcgabreca_rna_seq_hg38_r35
WHERE protein_coding_gene = 'Yes'
  AND project_id = 'TCGA-BRCA'
  AND sample_type IN ('Solid Tissue Normal', 'Other Tissue Types')
GROUP BY case_id;
```

This SQL query selects all columns from the `tcgabreca_rna_seq_hg38_r35` table where the protein coding gene is identified, the project ID is TCGA-BRCA, and the sample type includes either "Solid Tissue Normal" or other tissue types. It groups the results by `case_id` to ensure that each case is represented only once in the output. Adjust the column names and conditions as necessary based on your actual database schema.