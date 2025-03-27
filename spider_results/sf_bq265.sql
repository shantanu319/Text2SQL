```sql
SELECT u.email
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE YEAR(o.created_at) = 2019 AND YEAR(o.updated_at) = 2019
GROUP BY u.id, u.email
ORDER BY SUM(o.quantity * o.sale_price) / COUNT(o.id) DESC
LIMIT 10;
```