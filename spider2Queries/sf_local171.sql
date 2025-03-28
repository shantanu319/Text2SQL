```sql
SELECT 
    EXTRACT(YEAR FROM date_of_birth) - EXTRACT(YEAR FROM first_term_date) AS years_elapsed,
    COUNT(DISTINCT legislator_id) AS distinct_legislators
FROM 
    legislators
WHERE 
    gender = 'Male' AND state = 'Louisiana'
    AND date_of_birth >= DATE_SUB(CURDATE(), INTERVAL 30 YEAR)
    AND date_of_birth < DATE_SUB(CURDATE(), INTERVAL 50 YEAR)
GROUP BY 
    years_elapsed;
```