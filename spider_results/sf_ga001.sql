```sql
SELECT 
    p.product_name, 
    SUM(p.quantity) AS total_quantity
FROM 
    purchases pu
JOIN 
    products p ON pu.product_id = p.product_id
WHERE 
    pu.purchase_date BETWEEN '2020-12-01' AND '2020-12-31'
    AND pu.item_name = 'Google Navy Speckled Tee'
GROUP BY 
    p.product_name
ORDER BY 
    total_quantity DESC
LIMIT 1;
```