```sql
SELECT hub_id, 
       SUM(CASE WHEN order_date BETWEEN '2023-02-01' AND '2023-02-28' THEN 1 ELSE 0 END) AS february_orders,
       SUM(CASE WHEN order_date BETWEEN '2023-03-01' AND '2023-03-31' THEN 1 ELSE 0 END) AS march_orders
FROM orders
GROUP BY hub_id
HAVING (march_orders - february_orders) / february_orders > 0.2;
```