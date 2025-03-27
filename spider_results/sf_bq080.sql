```sql
SELECT 
    DATE_TRUNC('day', block_timestamp) AS day,
    SUM(CASE WHEN trace_address IS NULL THEN 1 ELSE 0 END) AS external_contracts_created,
    SUM(CASE WHEN trace_address IS NOT NULL THEN 1 ELSE 0 END) AS internal_contracts_created,
    LAG(SUM(CASE WHEN trace_address IS NULL THEN 1 ELSE 0 END)) OVER (ORDER BY DATE_TRUNC('day', block_timestamp)) AS previous_external_contracts_created,
    LAG(SUM(CASE WHEN trace_address IS NOT NULL THEN 1 ELSE 0 END)) OVER (ORDER BY DATE_TRUNC('day', block_timestamp)) AS previous_internal_contracts_created
FROM 
    ethereum_traces
WHERE 
    block_timestamp BETWEEN '2018-08-30' AND '2018-09-30'
GROUP BY 
    DATE_TRUNC('day', block_timestamp)
ORDER BY 
    DATE_TRUNC('day', block_timestamp);
```

This query calculates the daily cumulative count of smart contracts created by external users and those created by other contracts within the specified date range. It uses `DATE_TRUNC` to ensure that each row represents a single day, and `LAG` to calculate the