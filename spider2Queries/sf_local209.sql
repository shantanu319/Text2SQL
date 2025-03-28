```sql
SELECT 
    s.store_name,
    COUNT(o.order_id) AS total_orders,
    SUM(CASE WHEN d.status = 'DELIVERED' THEN 1 ELSE 0 END) / COUNT(o.order_id) AS delivered_ratio
FROM 
    orders o
JOIN 
    stores s ON o.store_id = s.store_id
LEFT JOIN 
    deliveries d ON o.order_id = d.order_id
GROUP BY 
    s.store_name
ORDER BY 
    total_orders DESC
LIMIT 1;
```