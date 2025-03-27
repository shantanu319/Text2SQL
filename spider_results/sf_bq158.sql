```sql
SELECT histological_type, COUNT(*) AS mutation_count
FROM pan_can_atlas
WHERE gene = 'CDH1'
GROUP BY histological_type
ORDER BY mutation_count DESC
LIMIT 5;
```