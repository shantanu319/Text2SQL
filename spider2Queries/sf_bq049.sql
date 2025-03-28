```sql
SELECT 
    DATE_FORMAT(sale_date, '%Y-%m') AS month,
    SUM(quantity) / COUNT(DISTINCT customer_id) AS per_capita_sales
FROM 
    sales_data
WHERE 
    sale_date BETWEEN '2022-01-01' AND '2022-12-31'
    AND county = 'Dubuque'
    AND zip_code IN (
        SELECT 
            zip_code
        FROM 
            sales_data
        WHERE 
            product_name = 'Bourbon Whiskey'
        GROUP BY 
            zip_code
        ORDER BY 
            SUM(quantity) DESC
        LIMIT 2
    )
    AND age >= 21
GROUP BY 
    month;
```