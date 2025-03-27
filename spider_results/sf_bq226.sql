```sql
SELECT sender_address
FROM transactions
WHERE to_address IS NOT NULL
  AND block_size > 4096
  AND transaction_time >= '2023-01-01'
GROUP BY sender_address
ORDER BY COUNT(*) DESC
LIMIT 1;
```