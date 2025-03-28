To convert the given text into an SQL query, we need to identify the relevant tables and columns. Assuming we have a table named `actors` with columns `actor_id`, `first_appearance_year`, and `last_appearance_year`, the query can be constructed as follows:

```sql
SELECT COUNT(actor_id) AS num_actors_no_gap_longer_than_three_years
FROM (
    SELECT 
        actor_id,
        MIN(first_appearance_year) AS min_first_appearance,
        MAX(last_appearance_year) AS max_last_appearance
    FROM 
        actors
    GROUP BY 
        actor_id
) AS actor_ranges
WHERE 
    (max_last_appearance - min_first_appearance + 1) <= 3;
```

### Explanation:
1. **Subquery**: The subquery groups the actors by `actor_id` and calculates the minimum and maximum years they appeared in films.
2. **Main Query**: The main query counts the number of actors whose range of appearances (from the earliest to the latest year) is 3 or fewer years long.

This query assumes that the `first_appearance_year` and `last_appearance_year` are integers representing the years. If these fields are stored differently (e.g., as strings