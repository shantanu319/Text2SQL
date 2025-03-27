```sql
SELECT 
    tx_hash AS "Transaction Hash",
    CONCAT('https://etherscan.io/tx/', tx_hash) AS "Etherscan Link",
    l1_token_address AS "L1 Token Address",
    l2_token_address AS "L2 Token Address",
    sender_address AS "Sender Address",
    receiver_address AS "Receiver Address",
    CAST(SUBSTRING(deposited_amount, 3) AS UNSIGNED) AS "Deposited Amount"
FROM 
    deposit_finalized_events
WHERE 
    block_number = 29815485
    AND bridge_contract = '0xstandardbridgeoptimism'
    AND event_signature = '0x3303facd24627943a92e9dc87cfbb34b15c49b726eec3ad3487c16be9ab8efe8';
```

This SQL query retrieves all finalized deposits into Optimism at block 29815485 using the Optimism Standard Bridge, including transaction hash, an Etherscan link, L1 and L2 token addresses, sender and receiver addresses with leading zeroes stripped, and the deposited