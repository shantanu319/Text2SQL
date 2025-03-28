```sql
SELECT p.product_name
FROM (
    SELECT 
        s.product_id,
        s.month,
        AVG(s.sales) OVER (ORDER BY s.month ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS adjusted_sales
    FROM 
        sales_data s
    WHERE 
        s.year = 2017
        AND s.month >= 1
        AND s.month <= 12
) AS adjusted_sales_data
JOIN 
    products p ON adjusted_sales_data.product_id = p.product_id
WHERE 
    adjusted_sales > 2;
```