```sql
SELECT 
    traffic_source,
    SUM(product_revenue) AS total_product_revenue,
    MAX(DAILY_REVENUE) AS max_daily_revenue,
    MAX(WEEKLY_REVENUE) AS max_weekly_revenue,
    MAX(MONTHLY_REVENUE) AS max_monthly_revenue
FROM 
    product_traffic
WHERE 
    YEAR = 2017 AND QUARTER = 'First Half'
GROUP BY 
    traffic_source
ORDER BY 
    total_product_revenue DESC
LIMIT 1;
```

This SQL query calculates the total product revenue for each traffic source during the first half of 2017. It then finds the traffic source with the highest total product revenue and retrieves the maximum daily, weekly, and monthly product revenues for that source over the specified period. The results are ordered by total product revenue in descending order and limited to the top-performing source. Adjust the table and column names as necessary to fit your actual database schema.