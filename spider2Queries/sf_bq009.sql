```sql
SELECT 
    traffic_source,
    SUM(transaction_revenue) AS total_transaction_revenue,
    MAX(MONTHLY_TRANSACTION_REVENUE) - MIN(MONTHLY_TRANSACTION_REVENUE) AS revenue_difference
FROM 
    transactions
WHERE 
    YEAR = 2017
GROUP BY 
    traffic_source
ORDER BY 
    total_transaction_revenue DESC
LIMIT 1;
```

This query calculates the total transaction revenue for each traffic source in 2017, finds the maximum and minimum monthly transaction revenue for each traffic source, and then determines the difference between these values. The result is ordered by total transaction revenue in descending order and limited to the top result. Note that `MONTHLY_TRANSACTION_REVENUE` should be replaced with the actual column name that contains the monthly transaction revenue data. Also, ensure that the date columns are correctly formatted and accessible in your database schema.