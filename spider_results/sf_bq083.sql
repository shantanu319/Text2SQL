```sql
SELECT 
    DATE_TRUNC('day', t.block_timestamp) AS transaction_date,
    SUM(CASE WHEN t.input LIKE '0x40c10f19%' THEN hex_to_millions(t.input) ELSE -hex_to_millions(t.input) END) AS daily_change_in_usdc
FROM 
    ethereum_transactions t
WHERE 
    t.contract_address = '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48'
    AND EXTRACT(YEAR FROM t.block_timestamp) = 2023
GROUP BY 
    transaction_date
ORDER BY 
    daily_change_in_usdc DESC;
```

Note: The function `hex_to_millions` is assumed to be defined elsewhere in your database schema to handle the conversion from hexadecimal to millions. You may need to adjust or implement this function according to your specific requirements. Additionally, ensure that the `ethereum_transactions` table contains the necessary columns (`block_timestamp`, `contract_address`, `input`).