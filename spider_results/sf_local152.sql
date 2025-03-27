```sql
SELECT 
    director_id,
    director_name,
    COUNT(movie_id) AS num_movies,
    ROUND(AVG(TIME_TO_SEC(TIMEDIFF(next_movie_date, current_movie_date))) / 60, 0) AS avg_inter_movie_duration,
    AVG(rating) AS avg_rating,
    SUM(votes) AS total_votes,
    MIN(rating) AS min_rating,
    MAX(rating) AS max_rating,
    SUM(duration) AS total_movie_duration
FROM 
    movies
JOIN 
    directors ON movies.director_id = directors.director_id
LEFT JOIN 
    (
        SELECT 
            movie_id,
            LEAD(current_movie_date) OVER (PARTITION BY director_id ORDER BY current_movie_date) AS next_movie_date
        FROM 
            movies
    ) AS next_movies ON movies.movie_id = next_movies.movie_id
GROUP BY 
    director_id, director_name
ORDER BY 
    num_movies DESC, total_movie_duration DESC;
```