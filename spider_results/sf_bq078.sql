```sql
SELECT T1.approved_symbol
FROM IMPC.target AS T1
JOIN IMPC.gene_association AS T2 ON T1.id = T2.target_id
WHERE T2.disease_id = 'EFO_0000676'
ORDER BY T2.overall_score DESC
LIMIT 1;
```