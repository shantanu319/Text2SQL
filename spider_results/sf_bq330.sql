```sql
SELECT z.zip_code, AVG(b.block_group_count) AS avg_block_group_count
FROM zip_codes z
JOIN block_groups b ON z.zip_code = b.zip_code
GROUP BY z.zip_code
ORDER BY avg_block_group_count DESC
LIMIT 1;
```