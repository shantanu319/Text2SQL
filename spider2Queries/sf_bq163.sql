```sql
SELECT gene_name, AVG(x_value) AS avg_x_value
FROM msk_sclc_patients
WHERE cluster = 41 AND age = 74
GROUP BY gene_name
ORDER BY avg_x_value DESC
LIMIT 20;
```