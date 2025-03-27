```sql
SELECT 
    MAX(balance) AS highest_balance,
    MIN(balance) AS lowest_balance
FROM (
    SELECT 
        address,
        SUM(amount) AS balance
    FROM 
        dash_transactions
    WHERE 
        block_timestamp >= '2023-04-01' AND 
        block_timestamp < '2023-05-01'
    GROUP BY 
        address
) AS transactions;
```