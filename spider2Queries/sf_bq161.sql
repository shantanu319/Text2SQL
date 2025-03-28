```sql
SELECT 
    SUM(CASE WHEN KRAS_Mutation AND TP53_Mutation THEN 1 ELSE 0 END) AS WithMutations,
    SUM(CASE WHEN NOT KRAS_Mutation AND NOT TP53_Mutation THEN 1 ELSE 0 END) AS WithoutMutations,
    (SUM(CASE WHEN KRAS_Mutation AND TP53_Mutation THEN 1 ELSE 0 END) - SUM(CASE WHEN NOT KRAS_Mutation AND NOT TP53_Mutation THEN 1 ELSE 0 END)) AS NetDifference
FROM 
    TCGA_Paad_Data
WHERE 
    Mutation_Quality_Filter = 'Passed';
```

This SQL query calculates the net difference between the number of PAAD patients with mutations in both KRAS and TP53 genes and those without any mutations in these genes. It uses a `CASE` statement within the `SUM` function to count patients with specific mutations and then subtracts the count of patients without any mutations from the count of patients with mutations. The results are filtered by ensuring that only patients whose mutations have passed quality filters are considered.