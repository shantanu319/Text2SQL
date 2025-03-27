```sql
WITH RankedBlocks AS (
    SELECT 
        block_time,
        block_interval,
        ROW_NUMBER() OVER (ORDER BY block_time) AS rn
    FROM 
        bitcoin_blocks
    WHERE 
        block_time >= '2023-01-01' AND block_time < '2024-01-01'
),
AdjustedIntervals AS (
    SELECT 
        rb.block_time,
        rb.block_interval,
        rb.rn,
        LAG(rb.block_time) OVER (PARTITION BY DATE(rb.block_time) ORDER BY rb.rn) AS prev_block_time,
        LEAD(rb.block_time) OVER (PARTITION BY DATE(rb.block_time) ORDER BY rb.rn) AS next_block_time
    FROM 
        RankedBlocks rb
)
SELECT 
    DATE(block_time) AS date,
    AVG(block_interval) AS avg_block_interval
FROM 
    AdjustedIntervals
WHERE 
    prev_block_time IS NOT NULL AND next_block_time IS NOT NULL
GROUP BY 
    DATE(block_time)
ORDER BY 
    avg_block_interval DESC
LIMIT 10;
```

This query calculates the daily average Bitcoin block interval for 2023 by joining consecutive blocks using row-numbered