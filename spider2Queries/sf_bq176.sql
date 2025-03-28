```sql
SELECT CASE_BARCODE
FROM TCGA_LAML
WHERE cytoband = '15q11'
AND chromosome = 15
ORDER BY weighted_average_copy_number DESC
LIMIT 1;
```