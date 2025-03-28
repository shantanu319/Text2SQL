```sql
SELECT 
    MAX(net_change) AS max_net_change,
    MIN(net_change) AS min_net_change
FROM (
    SELECT 
        SUM(CASE WHEN tx_type = 'debit' THEN value ELSE 0 END) - 
        SUM(CASE WHEN tx_type = 'credit' THEN value ELSE 0 END) + 
        SUM(gas_used * gas_price) AS net_change
    FROM 
        ethereum_classic_transactions
    WHERE 
        address IN (
            SELECT DISTINCT address 
            FROM ethereum_classic_transactions 
            WHERE date = '2016-10-14'
        )
        AND status = 'success'
        AND tx_type IN ('debit', 'credit')
    GROUP BY 
        address
) AS net_changes;
```

This query calculates the net change in balances for each Ethereum Classic address on October 14, 2016. It sums up the values sent to addresses (debits) and received from addresses (credits), subtracts these amounts, and adds the total gas used multiplied by the gas price for both miners and senders. The results are then grouped by address and aggregated to find the maximum and minimum net changes across all addresses. Only successful transactions are considered. <