```sql
SELECT COUNT(DISTINCT actor_id) AS num_actors_with_shahrukh_number_2
FROM (
    SELECT DISTINCT actor1_id AS actor_id
    FROM film_actor
    WHERE actor2_id = (SELECT actor_id FROM film_actor WHERE actor_name = 'Shahrukh Khan')
    INTERSECT
    SELECT DISTINCT actor1_id AS actor_id
    FROM film_actor
    WHERE actor2_id != (SELECT actor_id FROM film_actor WHERE actor_name = 'Shahrukh Khan')
) AS subquery;
```