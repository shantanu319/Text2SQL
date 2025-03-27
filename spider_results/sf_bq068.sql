```sql
SELECT 
    CASE 
        WHEN address_type = 'S' THEN 'Single'
        ELSE 'Multi'
    END AS address_type,
    MAX(final_balance) AS max_final_balance,
    MIN(final_balance) AS min_final_balance
FROM (
    SELECT 
        address_type,
        SUM(CASE WHEN tx_type = 'input' THEN -value ELSE value END) AS final_balance
    FROM bitcoin_cash_transactions
    WHERE timestamp BETWEEN '2014-03-01' AND '2014-04-01'
    GROUP BY address_type
) AS balance_summary
GROUP BY address_type;
```