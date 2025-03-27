```sql
SELECT 
    s.store_id,
    YEAR(r.rental_date) AS rental_year,
    MONTH(r.rental_date) AS rental_month,
    COUNT(*) AS total_rentals
FROM 
    rental r
JOIN 
    store s ON r.store_id = s.store_id
GROUP BY 
    s.store_id, YEAR(r.rental_date), MONTH(r.rental_date)
ORDER BY 
    total_rentals DESC;
```