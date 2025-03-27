```sql
SELECT 
    p.product_name,
    o.order_month,
    SUM(o.sale_price) - SUM(o.cost_price) AS total_profit
FROM 
    orders o
JOIN 
    products p ON o.product_id = p.product_id
WHERE 
    o.order_date BETWEEN '2019-01-01' AND '2022-08-31'
    AND o.status NOT IN ('canceled', 'returned')
GROUP BY 
    p.product_name, o.order_month
ORDER BY 
    p.product_name, o.order_month, total_profit DESC
LIMIT 3;
```

This SQL query retrieves the names of the top three most profitable products for each month from January 2019 through August 2022, excluding any products associated with orders that were canceled or returned. The profit is calculated by subtracting the sum of the costs of sold items from the sum of the sale prices of all order items in that month. The results are ordered by product name and then by month, and only the top three profits per product are returned.