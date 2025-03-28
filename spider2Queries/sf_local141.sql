```sql
SELECT 
    salesperson_id,
    YEAR(sales_date) AS year,
    (total_sales - sales_quota) AS sales_difference
FROM 
    sales_data
GROUP BY 
    salesperson_id, YEAR(sales_date)
ORDER BY 
    salesperson_id, YEAR(sales_date);
```