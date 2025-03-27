```sql
SELECT 
    o.product_name, 
    SUM(o.quantity) AS total_quantity
FROM 
    orders o
JOIN 
    order_items oi ON o.order_id = oi.order_id
JOIN 
    products p ON oi.product_id = p.product_id
WHERE 
    p.product_name = 'Google Red Speckled Tee'
    AND o.purchase_date >= DATE_TRUNC('month', '2020-11-01')
    AND o.purchase_date < DATE_TRUNC('month', '2021-04-01')
GROUP BY 
    o.product_name
ORDER BY 
    total_quantity DESC;
```