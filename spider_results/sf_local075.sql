```sql
SELECT 
    p.product_id,
    COUNT(CASE WHEN event_type = 'view' THEN 1 ELSE NULL END) AS view_count,
    COUNT(CASE WHEN event_type = 'add_to_cart' THEN 1 ELSE NULL END) AS add_to_cart_count,
    COUNT(CASE WHEN event_type = 'leave_cart' THEN 1 ELSE NULL END) AS leave_cart_count,
    COUNT(CASE WHEN event_type = 'purchase' THEN 1 ELSE NULL END) AS purchase_count
FROM 
    events e
JOIN 
    products p ON e.product_id = p.product_id
WHERE 
    p.page_id NOT IN (1, 2, 12, 13)
GROUP BY 
    p.product_id;
```