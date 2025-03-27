```sql
SELECT c.first_name, a.amount_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN albums a ON oi.album_id = a.album_id
WHERE a.artist_id = (SELECT artist_id FROM albums WHERE sales = (SELECT MAX(sales) FROM albums))
  AND a.amount_spent < 1;
```