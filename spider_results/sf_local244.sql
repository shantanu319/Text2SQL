```sql
SELECT 
    CASE 
        WHEN DURATION < 60 THEN 'Short'
        WHEN DURATION BETWEEN 60 AND 120 THEN 'Medium'
        ELSE 'Long'
    END AS CATEGORY,
    MIN(DURATION / 60) AS MIN_TIME_MINUTES,
    MAX(DURATION / 60) AS MAX_TIME_MINUTES,
    SUM(REVENUE) AS TOTAL_REVENUE
FROM 
    TRACKS
GROUP BY 
    CATEGORY;
```

This SQL query calculates the duration of each track, classifies them into short, medium, or long categories based on their duration in seconds, and then outputs the minimum and maximum time for each category in minutes. It also calculates the total revenue for each category and groups the results by category. The `REVENUE` column is assumed to be present in the `TRACKS` table, representing the revenue generated from each track. Adjust the column names if they are different in your database schema.