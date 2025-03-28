```sql
SELECT 
    t1.tx_hash AS "From",
    t2.tx_hash AS "Intermediate",
    t3.tx_hash AS "To"
FROM 
    zilliqa_transactions t1
JOIN 
    zilliqa_transactions t2 ON t1.to_address = t2.from_address AND t2.block_timestamp > t1.block_timestamp
JOIN 
    zilliqa_transactions t3 ON t2.to_address = t3.from_address AND t3.block_timestamp > t2.block_timestamp
WHERE 
    t1.from_address = 'zil1jrpjd8pjuv50cfkfr7eu6yrm3rn5u8rulqhqpz'
    AND t3.to_address = 'zil19nmxkh020jnequql9kvqkf3pkwm0j0spqtd26e'
    AND t1.confirmation_status = 'Confirmed'
    AND t2.confirmation_status = 'Confirmed'
    AND t3.confirmation_status = 'Confirmed'
    AND (
        SELECT COUNT(*) FROM zilliqa_transactions WHERE from_address = t2.to_address AND block_timestamp > t2.block_timestamp
        ) <= 50
ORDER BY 
    t1