```sql
SELECT 
    TIMESTAMP(block_timestamp) AS blockchain_timestamp,
    block_number,
    tx_hash
FROM 
    ethereum_logs
WHERE 
    address = '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8'
    AND (topic_0 = '0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde' OR topic_0 = '0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c')
ORDER BY 
    block_timestamp ASC
LIMIT 5;
```