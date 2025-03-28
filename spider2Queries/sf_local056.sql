```sql
SELECT c.full_name
FROM customers c
JOIN payments p ON c.customer_id = p.customer_id
GROUP BY c.customer_id, c.full_name
ORDER BY AVG(p.amount_change) DESC
LIMIT 1;
```