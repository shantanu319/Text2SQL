To convert the given text into an SQL query, we need to consider the following steps:

1. **Identify the relevant columns**: We assume there are columns like `movie_release_year` and `film_count`.

2. **Determine the logic**: We need to find the ten-year period starting from each unique movie release year and calculate the total number of films for each period.

3. **Output the required information**: We need to output the start year and the total count for each period.

Here's how you can write the SQL query:

```sql
SELECT 
    movie_release_year AS start_year,
    COUNT(*) AS total_count
FROM 
    your_table_name
GROUP BY 
    (movie_release_year - 1) / 10 * 10
ORDER BY 
    total_count DESC;
```

### Explanation:

- **`(movie_release_year - 1) / 10 * 10`**: This expression calculates the start year of the ten-year period by rounding down the movie release year to the nearest multiple of 10.
  
- **`COUNT(*)`**: This counts the number of films released within each ten-year period.

- **`GROUP BY ... ORDER BY ...`**: This groups the results by the