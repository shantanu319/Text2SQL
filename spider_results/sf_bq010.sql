```sql
SELECT p.product_name, SUM(o.quantity) AS total_quantity_sold
FROM orders o
JOIN products p ON o.product_id = p.product_id
WHERE o.customer_id IN (
    SELECT customer_id
    FROM orders
    WHERE order_date BETWEEN '2017-07-01' AND '2017-07-31'
      AND product_name = 'Youtube Men’s Vintage Henley'
)
AND p.product_name != 'Youtube Men’s Vintage Henley'
GROUP BY p.product_name
ORDER BY total_quantity_sold DESC
LIMIT 1;
```