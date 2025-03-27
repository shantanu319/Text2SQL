```sql
SELECT sample_id, COUNT(*) AS count_of_alternate_alleles
FROM genome_data
WHERE genotype = '00' AND alternate_allele_count = 1
GROUP BY sample_id
ORDER BY count_of_alternate_alleles DESC
LIMIT 10;
```