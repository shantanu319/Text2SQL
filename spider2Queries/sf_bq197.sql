```sql
SELECT 
    DATE_TRUNC('month', o.order_date) AS month,
    p.product_name,
    p.brand,
    c.category,
    SUM(o.quantity * p.price) AS total_sales,
    SUM(o.quantity * p.price * (1 - p.discount)) AS total_revenue,
    o.status
FROM 
    orders o
JOIN 
    products p ON o.product_id = p.product_id
JOIN 
    categories c ON p.category_id = c.category_id
WHERE 
    o.status = 'Complete'
    AND p.brand IS NOT NULL
    AND DATE_TRUNC('year', o.order_date) < 2024
GROUP BY 
    DATE_TRUNC('month', o.order_date),
    p.product_name,
    p.brand,
    c.category,
    o.status
ORDER BY 
    month DESC;
```

This SQL query retrieves the required information by joining the `orders`, `products`, and `categories` tables. It filters orders based on their status being 'Complete' and products having a non-null brand. The results are grouped by month, product details, and order status, and ordered by month in descending order to show the most recent data first. The query calculates total sales and total revenue for each product within