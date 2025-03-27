```sql
SELECT 
    p.product_name,
    m.month_name,
    p.retail_price - p.cost AS profit
FROM 
    products p
JOIN 
    sales s ON p.product_id = s.product_id
JOIN 
    months m ON s.sale_date BETWEEN m.start_date AND m.end_date
WHERE 
    YEAR(s.sale_date) = 2020
GROUP BY 
    p.product_name, m.month_name
ORDER BY 
    m.month_number;
```