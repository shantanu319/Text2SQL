```sql
SELECT 
    ticker,
    DATE(date) AS date,
    (CASE 
        WHEN volume LIKE '%K' THEN CAST(SUBSTRING(volume, 1, LENGTH(volume) - 1) AS DECIMAL) * 1000
        WHEN volume LIKE '%M' THEN CAST(SUBSTRING(volume, 1, LENGTH(volume) - 1) AS DECIMAL) * 1000000
        ELSE CASE 
            WHEN volume = '-' THEN 0
            ELSE CAST(volume AS DECIMAL)
        END
    END) AS adjusted_volume,
    LAG(CASE 
        WHEN volume LIKE '%K' THEN CAST(SUBSTRING(volume, 1, LENGTH(volume) - 1) AS DECIMAL) * 1000
        WHEN volume LIKE '%M' THEN CAST(SUBSTRING(volume, 1, LENGTH(volume) - 1) AS DECIMAL) * 1000000
        ELSE CASE 
            WHEN volume = '-' THEN 0
            ELSE CAST(volume AS DECIMAL)
        END
    END) OVER (PARTITION BY ticker ORDER BY date) AS prev_day_adjusted_volume,
    (adjusted_volume / prev_day_adjusted_volume