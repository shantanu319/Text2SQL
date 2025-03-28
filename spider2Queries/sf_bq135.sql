```sql
SELECT DATE, SUM(TransactionAmount) AS TotalTransactionAmount
FROM ZilliqaBlockchainData
WHERE DATE < '2022-01-01'
GROUP BY DATE
ORDER BY TotalTransactionAmount DESC
LIMIT 1;
```