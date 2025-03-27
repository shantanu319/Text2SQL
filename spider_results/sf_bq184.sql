```sql
SELECT 
    DATE_TRUNC('day', timestamp_column) AS date,
    SUM(CASE WHEN trace_address IS NULL THEN 1 ELSE 0 END) OVER (ORDER BY DATE_TRUNC('day', timestamp_column)) AS external_cumulative_count,
    SUM(CASE WHEN trace_address IS NOT NULL THEN 1 ELSE 0 END) OVER (ORDER BY DATE_TRUNC('day', timestamp_column)) AS internal_cumulative_count
FROM 
    your_table_name
WHERE 
    timestamp_column BETWEEN '2017-01-01' AND '2021-12-31'
GROUP BY 
    DATE_TRUNC('day', timestamp_column)
ORDER BY 
    date;
```

This query calculates the daily cumulative counts of smart contracts created by external addresses and those created by other contracts for each date within the specified range. It uses window functions to maintain a running total for both categories, ensuring that all dates in the range are included, even if no new contracts were created on some days. The `DATE_TRUNC` function is used to group the data by day, and the `SUM` function with a `CASE` statement is used to count the number of contracts for each category. The results are ordered by date. Replace `