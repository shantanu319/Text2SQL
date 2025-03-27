```sql
SELECT 
    e.employee_id,
    COUNT(o.order_id) AS late_orders_count,
    (COUNT(CASE WHEN o.shipped_date >= o.required_date THEN 1 ELSE NULL END) * 100.0 / COUNT(o.order_id)) AS late_order_percentage
FROM 
    employees e
JOIN 
    orders o ON e.employee_id = o.employee_id
WHERE 
    e.total_orders > 50
GROUP BY 
    e.employee_id
ORDER BY 
    late_order_percentage DESC
LIMIT 3;
```