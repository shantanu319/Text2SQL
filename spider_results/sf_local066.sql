```sql
SELECT 
    i.name AS ingredient_name,
    SUM(p.quantity) AS total_quantity
FROM 
    pizza_orders p
JOIN 
    ingredients i ON p.ingredient_id = i.id
GROUP BY 
    i.name;
```