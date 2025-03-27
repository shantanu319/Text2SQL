```sql
SELECT 
    d.division_name,
    AVG(h.quantity_sold) AS avg_quantity_sold
FROM 
    divisions d
JOIN 
    sales s ON d.division_id = s.division_id
JOIN 
    hardware h ON s.hardware_id = h.hardware_id
WHERE 
    YEAR(s.sale_date) = 2021
GROUP BY 
    d.division_name
ORDER BY 
    SUM(h.quantity_sold) DESC
LIMIT 3;
```