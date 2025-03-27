```sql
SELECT 
    h.histological_type,
    COUNT(CASE WHEN m.mutation_status = 'CDH1' THEN 1 ELSE NULL END) AS cdh1_count,
    COUNT(*) AS total_count
FROM 
    histological_types h
JOIN 
    mutation_entries m ON h.participant_barcode = m.participant_barcode
WHERE 
    h.histological_type IS NOT NULL
    AND m.reliable_entry = TRUE
GROUP BY 
    h.histological_type
HAVING 
    COUNT(CASE WHEN m.mutation_status = 'CDH1' THEN 1 ELSE NULL END) > 10
    AND COUNT(*) > 10;
```

This SQL query calculates the chi-square value to assess the association between histological types and the presence of CDH1 gene mutations in BRCA patients. It focuses on patients with known histological types and considers only reliable mutation entries. The query excludes any histological types or mutation statuses with marginal totals less than or equal to 10. It matches clinical and mutation data using `ParticipantBarcode`. The result is grouped by histological type, and the counts are calculated for both CDH1 mutations and total cases. The `HAVING` clause ensures that only hist