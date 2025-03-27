```sql
SELECT COUNT(T1.state) AS num_top_states
FROM (
    SELECT state, 
           SUM(CASE WHEN alien_type = 'friendly' THEN 1 ELSE 0 END) / COUNT(*) AS friendly_percentage,
           AVG(alien_age) AS avg_alien_age
    FROM alien_data
    GROUP BY state
    ORDER BY friendly_percentage DESC, avg_alien_age DESC
    LIMIT 10
) T1
WHERE T1.friendly_percentage > 50 AND T1.avg_alien_age > 200;
```