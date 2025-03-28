```sql
WITH TopGenres AS (
    SELECT g.genre_name, COUNT(*) AS movie_count
    FROM genre g
    JOIN movie m ON g.genre_id = m.genre_id
    WHERE m.rating > 8
    GROUP BY g.genre_name
    ORDER BY movie_count DESC
    LIMIT 3
),
TopDirectors AS (
    SELECT d.director_name, COUNT(*) AS film_count
    FROM director d
    JOIN movie m ON d.director_id = m.director_id
    WHERE m.rating > 8
    AND m.genre_id IN (SELECT genre_id FROM TopGenres)
    GROUP BY d.director_name
    ORDER BY film_count DESC
    LIMIT 4
)
SELECT td.director_name, td.film_count
FROM TopDirectors td;
```