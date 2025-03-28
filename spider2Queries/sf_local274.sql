```sql
SELECT p.product_name, AVG(picked_units) AS average_picked_units
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
WHERE oi.order_id = 421
GROUP BY p.product_name;
```