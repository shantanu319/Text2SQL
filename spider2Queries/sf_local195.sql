```sql
SELECT 
    COUNT(DISTINCT customer_id) AS total_customers,
    COUNT(DISTINCT film_id) AS total_films_rented,
    (COUNT(DISTINCT film_id) / COUNT(DISTINCT customer_id)) * 100 AS percentage_of_customers_with_top_five_actors
FROM 
    rentals
WHERE 
    actor_id IN (
        SELECT 
            actor_id
        FROM 
            top_five_actors
    );
```

This SQL query calculates the percentage of customers who have rented films featuring the top five actors by counting the distinct customer IDs and film IDs in the `rentals` table, then dividing the number of films rented by the number of unique customers and multiplying by 100 to get a percentage. The subquery selects the `actor_id`s from the `top_five_actors` table to filter the rentals accordingly.