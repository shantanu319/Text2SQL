```sql
SELECT 
    c.customer_unique_id,
    AVG(o.payment_value) AS avg_payment_value,
    o.city,
    o.state
FROM 
    customers c
JOIN 
    orders o ON c.customer_unique_id = o.customer_unique_id
GROUP BY 
    c.customer_unique_id, o.city, o.state
ORDER BY 
    COUNT(o.order_id) DESC
LIMIT 3;
```