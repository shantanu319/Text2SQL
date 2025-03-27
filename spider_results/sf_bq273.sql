```sql
SELECT 
    MONTH(order_date) AS month,
    SUM(sales - costs) AS total_profit
FROM 
    completed_orders
WHERE 
    order_date BETWEEN '2022-08-01' AND '2023-11-30'
GROUP BY 
    MONTH(order_date)
ORDER BY 
    total_profit DESC
LIMIT 5;
```