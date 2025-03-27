```sql
SELECT 
    DATE_FORMAT(o.creation_date, '%Y-%m') AS month,
    SUM(i.quantity * p.price) AS total_sales,
    SUM(i.quantity * p.cost) AS total_cost,
    COUNT(o.order_id) AS total_complete_orders,
    (SUM(i.quantity * p.price) - SUM(i.quantity * p.cost)) AS total_profit,
    (SUM(i.quantity * p.price) - SUM(i.quantity * p.cost)) / SUM(i.quantity * p.cost) AS profit_to_cost_ratio
FROM 
    orders o
JOIN 
    order_items i ON o.order_id = i.order_id
JOIN 
    products p ON i.product_id = p.product_id
WHERE 
    o.category = 'Sleep & Lounge'
    AND o.status = 'Complete'
    AND o.creation_date BETWEEN '2023-01-01' AND '2023-12-31'
GROUP BY 
    DATE_FORMAT(o.creation_date, '%Y-%m');
```