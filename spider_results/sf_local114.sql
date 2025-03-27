```sql
SELECT 
    r.region_name,
    COUNT(o.order_id) AS order_count,
    SUM(o.total_amount) AS total_sales_amount,
    sr.sales_rep_name,
    sr.total_sales_amount AS rep_total_sales_amount
FROM 
    regions r
JOIN 
    sales_regions sr ON r.region_id = sr.region_id
JOIN 
    orders o ON sr.sales_region_id = o.sales_region_id
GROUP BY 
    r.region_name, sr.sales_rep_name, sr.total_sales_amount
ORDER BY 
    r.region_name, sr.total_sales_amount DESC;
```

This SQL query provides a detailed web sales report for each region by joining the `regions`, `sales_regions`, and `orders` tables. It calculates the number of orders, total sales amount, and the name and sales amount of all sales representatives who achieved the highest total sales amount in that region. In case of a tie, it includes all representatives with the same highest total sales amount. The results are grouped by region and sorted by total sales amount in descending order.