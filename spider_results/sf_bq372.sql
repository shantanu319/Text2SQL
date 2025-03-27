```sql
WITH LostOrders AS (
    SELECT 
        o.customer_category,
        MAX(o.order_value) - i.invoice_value AS lost_order_value
    FROM 
        Orders o
    LEFT JOIN 
        Invoices i ON o.order_id = i.order_id
    WHERE 
        i.invoice_id IS NULL
    GROUP BY 
        o.customer_category
),
AverageLostOrderValue AS (
    SELECT 
        AVG(lost_order_value) AS avg_lost_order_value
    FROM 
        LostOrders
)
SELECT 
    l.customer_category, 
    l.lost_order_value
FROM 
    LostOrders l
JOIN 
    AverageLostOrderValue alv ON 1 = 1
WHERE 
    l.lost_order_value = (SELECT MIN(abs(lv.avg_lost_order_value - l.lost_order_value)) FROM LostOrders lv);
```

This SQL query first calculates the maximum lost order value for each customer category among those orders that do not appear in the invoice table. It then determines the average of these maximum lost order values. Finally, it identifies the customer category whose maximum lost order value is closest to the overall average. The `LEFT JOIN` ensures that only orders without an associated invoice are considered. The `ABS` function is used