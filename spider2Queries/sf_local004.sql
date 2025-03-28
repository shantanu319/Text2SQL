```sql
SELECT 
    COUNT(*) AS total_orders,
    AVG(payment) AS avg_payment_per_order,
    MAX(DATEDIFF(LATEST_PURCHASE_DATE, EARLIEST_PURCHASE_DATE)) / 7 AS customer_lifespan_in_weeks
FROM (
    SELECT 
        customer_id,
        SUM(payment) AS payment,
        MIN(purchase_date) AS EARLIEST_PURCHASE_DATE,
        MAX(purchase_date) AS LATEST_PURCHASE_DATE
    FROM 
        orders
    GROUP BY 
        customer_id
) AS subquery
ORDER BY 
    avg_payment_per_order DESC
LIMIT 3;
```