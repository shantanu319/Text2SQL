```sql
SELECT COUNT(DISTINCT c.customer_id) AS matching_customers
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN invoices i ON c.customer_id = i.customer_id
WHERE (
    SELECT COUNT(*)
    FROM order_items oi1
    JOIN invoice_items ii1 ON oi1.order_item_id = ii1.order_item_id
    WHERE oi1.order_id = o.order_id AND ii1.invoice_id = i.invoice_id
) > 0
AND (
    SELECT COUNT(*)
    FROM order_items oi2
    JOIN invoice_items ii2 ON oi2.order_item_id = ii2.order_item_id
    WHERE oi2.order_id = o.order_id AND ii2.invoice_id = i.invoice_id
) = (
    SELECT COUNT(*)
    FROM order_items oi3
    JOIN invoice_items ii3 ON oi3.order_item_id = ii3.order_item_id
    WHERE oi3.order_id = o.order_id AND ii3.invoice_id = i.invoice_id
)
AND (
    SELECT SUM(oi1.quantity * oi1.unit_price)
    FROM order_items oi1
    JOIN invoice_items ii1 ON oi1.order_item_id = ii1.order_item_id
    WHERE oi1