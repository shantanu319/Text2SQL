```sql
SELECT address
FROM (
    SELECT 
        t1.address,
        ABS(t1.balance - t2.balance) AS balance_difference
    FROM 
        ethereum_tokens t1
    JOIN 
        ethereum_tokens t2 ON t1.address = '0x0d8775f648430679a709e98d2b0cb6250d2887ef0'
    WHERE 
        t2.address = '0x1e15c05cbad367f044cbfbafda3d9a1510db5513'
    EXCEPT
    SELECT 
        '0x0000000000000000000000000000000000000000' AS address
)
ORDER BY 
    balance_difference DESC
LIMIT 6;
```