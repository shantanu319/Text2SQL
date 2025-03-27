Certainly! To achieve this, we need to write an SQL query that filters the movies based on the specified criteria and then determines which actor starred most frequently. Here's how you can structure the query:

```sql
SELECT T3.actor_name AS full_name, COUNT(*) AS frequency
FROM movie AS T1
JOIN movie_actor AS T2 ON T1.movie_id = T2.movie_id
JOIN actor AS T3 ON T2.actor_id = T3.actor_id
WHERE T1.category = 'English-language children\'s'
  AND T1.rating IN ('G', 'PG')
  AND T1.runtime <= 120
  AND YEAR(T1.release_date) BETWEEN 2000 AND 2010
GROUP BY T3.actor_name
ORDER BY frequency DESC
LIMIT 1;
```

### Explanation:
- **Tables**: We use `movie`, `movie_actor`, and `actor` tables to link movies with their actors.
- **Filters**:
  - `T1.category = 'English-language children\'s'`: Filters for movies in the specified category.
  - `T1.rating IN ('G', 'PG')`: Ensures the movies are rated either G or PG.
  - `T1