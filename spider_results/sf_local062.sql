```sql
SELECT 
    COUNT(*) AS customer_count,
    MIN(total_profit) AS min_total_profit,
    MAX(total_profit) AS max_total_profit
FROM (
    SELECT 
        c.customer_id,
        SUM(p.quantity_sold * (p.unit_price - p.unit_cost)) AS total_profit
    FROM 
        customers c
    JOIN 
        orders o ON c.customer_id = o.customer_id
    JOIN 
        order_items oi ON o.order_id = oi.order_id
    JOIN 
        products p ON oi.product_id = p.product_id
    WHERE 
        c.country = 'Italy' AND YEAR(o.order_date) = 2021 AND MONTH(o.order_date) = 12
    GROUP BY 
        c.customer_id
) AS grouped_customers
GROUP BY 
    FLOOR((total_profit - MIN(total_profit)) / (MAX(total_profit) - MIN(total_profit))) * 10;
```

This query first calculates the total profit for each Italian customer in December 2021. It then groups these customers into ten buckets based on their total profit using a floor function to map the profit to an interval. Finally, it provides the count of customers, the minimum and maximum total profits within each bucket