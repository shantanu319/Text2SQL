```sql
SELECT fc.category_name, SUM(r.rental_hours) AS total_rental_hours
FROM film_category fc
JOIN film f ON fc.film_id = f.film_id
JOIN inventory i ON f.film_id = i.film_id
JOIN rental r ON i.inventory_id = r.inventory_id
WHERE i.store_id IN (
    SELECT store_id FROM store WHERE address LIKE 'A%' OR address LIKE '%-%'
)
GROUP BY fc.category_name
ORDER BY total_rental_hours DESC
LIMIT 1;
```