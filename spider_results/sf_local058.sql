```sql
SELECT 
    segment_name,
    COUNT(DISTINCT product_id) AS unique_product_count_2020,
    (COUNT(DISTINCT product_id) - LAG(COUNT(DISTINCT product_id)) OVER (ORDER BY year DESC)) / NULLIF(LAG(COUNT(DISTINCT product_id)) OVER (ORDER BY year DESC), 0) * 100 AS percentage_increase
FROM 
    hardware_products
WHERE 
    year = 2020
GROUP BY 
    segment_name
ORDER BY 
    percentage_increase DESC;
```