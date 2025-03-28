```sql
SELECT SUM(balance / POWER(10, 18)) AS total_circulating_supply
FROM (
    SELECT 
        sender,
        receiver,
        amount_received - amount_sent AS balance
    FROM 
        transactions
    WHERE 
        sender != '0x000...' AND 
        receiver != '0x000...'
) AS filtered_transactions;
```