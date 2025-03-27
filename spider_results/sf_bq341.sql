```sql
SELECT t1.ethereum_address, SUM(t2.amount) AS total_balance
FROM transactions t1
JOIN transaction_details t2 ON t1.transaction_id = t2.transaction_id
WHERE t2.token_address = '0xa92a861fc11b99b24296af880011b47f9cafb5ab'
GROUP BY t1.ethereum_address
ORDER BY total_balance ASC
LIMIT 3;
```