```sql
SELECT 
    r.block_timestamp,
    r.request_id,
    s.symbol,
    CAST(r.rate AS DECIMAL) / r.multiplier AS adjusted_rate
FROM 
    oracle_requests r
JOIN 
    symbol_rates s ON r.script_id = 3 AND r.decoded_result LIKE '%' || s.symbol || '%'
ORDER BY 
    r.block_timestamp DESC
LIMIT 10;
```

This SQL query retrieves the 10 most recent Oracle requests with a specific script ID (3), extracts symbol-rate pairs from the decoded results, adjusts each rate by dividing it by the request's multiplier, and returns the block timestamp, request ID, symbol, and adjusted rate. The results are ordered chronologically by the block timestamp in descending order.