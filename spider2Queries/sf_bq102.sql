```sql
SELECT 
    start_position
FROM 
    variant_data
WHERE 
    gene_name = 'BRCA1'
    AND chromosome_number = 17
    AND reference_base = 'C'
    AND alternate_base = 'T'
    AND variant_type = 'missense_variant';
```