```sql
SELECT 
    pc.category_name,
    p.payment_type,
    COUNT(*) AS total_payments
FROM 
    products p
JOIN 
    product_categories pc ON p.product_category_id = pc.id
GROUP BY 
    pc.category_name, p.payment_type
ORDER BY 
    COUNT(*) DESC
LIMIT 3;
```