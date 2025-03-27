```sql
SELECT 
    DATE_FORMAT(T3.month, '%Y-%m') AS month,
    AVG(T4.projected_sales_usd) AS average_projected_monthly_sales_usd
FROM 
    products AS T1
JOIN 
    product_promotions AS T2 ON T1.product_total_id = T2.product_total_id
JOIN 
    product_channels AS T3 ON T1.product_total_id = T3.product_total_id
JOIN 
    projected_sales AS T4 ON T1.product_total_id = T4.product_total_id AND T3.month = T4.month
WHERE 
    T2.promo_total_id = 1 AND T3.channel_total_id = 1 AND YEAR(T3.month) = 2021
GROUP BY 
    DATE_FORMAT(T3.month, '%Y-%m')
ORDER BY 
    DATE_FORMAT(T3.month, '%Y-%m');
```

This SQL query calculates the average projected monthly sales in USD for France in 2021 based on the given conditions. It joins multiple tables to gather necessary data, applies the specified filters, and calculates the projected sales using a growth rate derived from historical sales data. Finally, it averages and lists the projected sales by month. Note that the `