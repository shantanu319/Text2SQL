```sql
SELECT gene_id, fpkm_uq_unstranded
FROM gene_expression_data
WHERE sample_type = 'Solid Tissue Normal'
  AND case_type = 'BRCA'
  AND sample_type IN ('Solid Tissue Normal')
ORDER BY fpkm_uq_unstranded DESC
LIMIT 5;
```