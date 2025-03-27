```sql
SELECT 
    t.address,
    MAX(t.tx_time) AS latest_tx_date,
    SUM(t.value) AS total_value
FROM 
    transactions t
WHERE 
    t.tx_time >= '2017-10-01' AND t.tx_time < '2017-11-01'
GROUP BY 
    t.address
HAVING 
    COUNT(DISTINCT t.tx_id) > 0
ORDER BY 
    latest_tx_date DESC, total_value DESC
LIMIT 1;
```