Certainly! Below is an SQL query that calculates the average single career span value in years for all baseball players based on the provided requirements:

```sql
WITH CareerSpans AS (
    SELECT 
        player_id,
        DATEDIFF(YEAR, debut_date, final_game_date) AS years,
        DATEDIFF(MONTH, debut_date, final_game_date) % 12 AS months,
        DATEDIFF(DAY, debut_date, final_game_date) % 365 AS days
    FROM 
        baseball_players
)
SELECT 
    ROUND(AVG(years + (months / 12) + (days / 365)), 2) AS average_career_span
FROM 
    CareerSpans;
```

### Explanation:
1. **Common Table Expression (CTE):** The `CareerSpans` CTE calculates the difference in years, months, and days between the debut and final game dates for each player.
   - `DATEDIFF(YEAR, debut_date, final_game_date)` computes the number of full years.
   - `DATEDIFF(MONTH, debut_date, final_game_date) % 12` computes the remaining months after accounting for full years.
   - `DATED