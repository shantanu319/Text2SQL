```sql
SELECT 
    actor_name,
    film_title,
    revenue_per_actor
FROM (
    SELECT 
        actor_name,
        film_title,
        SUM(revenue) AS total_revenue,
        COUNT(DISTINCT actor_id) AS num_actors
    FROM 
        films
    GROUP BY 
        actor_name, film_title
) AS film_revenues
JOIN (
    SELECT 
        actor_name,
        AVG(total_revenue / num_actors) AS revenue_per_actor
    FROM (
        SELECT 
            actor_name,
            SUM(revenue) AS total_revenue,
            COUNT(DISTINCT actor_id) AS num_actors
        FROM 
            films
        GROUP BY 
            actor_name
    ) AS actor_revenues
    GROUP BY 
        actor_name
) AS avg_revenues ON film_revenues.actor_name = avg_revenues.actor_name
ORDER BY 
    actor_name, revenue_per_actor DESC
LIMIT 3;
```

This SQL query calculates the top three revenue-generating films for each actor and the average revenue per actor in those films. It uses subqueries to first calculate the total revenue and number of actors for each film and then calculates the average revenue per actor for each actor. Finally, it orders