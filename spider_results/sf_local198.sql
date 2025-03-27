```sql
SELECT AVG(total_sales) AS median_total_sales
FROM (
    SELECT total_sales
    FROM sales_data
    WHERE country IN (
        SELECT country
        FROM sales_data
        GROUP BY country
        HAVING COUNT(customer_id) > 4
    )
    ORDER BY total_sales
    LIMIT (SELECT CEIL(COUNT(*) / 2.0) FROM sales_data)
    OFFSET (SELECT FLOOR((COUNT(*) - 1) / 2.0) FROM sales_data)
) AS subquery;
```