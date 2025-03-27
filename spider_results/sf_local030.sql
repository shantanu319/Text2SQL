```sql
SELECT 
    city,
    SUM(payment) AS total_payment,
    COUNT(*) AS total_delivered_orders
FROM 
    orders
WHERE 
    status = 'delivered'
GROUP BY 
    city
ORDER BY 
    total_payment ASC
LIMIT 5;

SELECT 
    AVG(total_payment) AS avg_total_payment,
    AVG(total_delivered_orders) AS avg_total_delivered_orders
FROM (
    SELECT 
        city,
        SUM(payment) AS total_payment,
        COUNT(*) AS total_delivered_orders
    FROM 
        orders
    WHERE 
        status = 'delivered'
    GROUP BY 
        city
    ORDER BY 
        total_payment ASC
    LIMIT 5
) AS subquery;
```