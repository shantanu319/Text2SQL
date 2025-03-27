```sql
SELECT 
    MONTH(T1.date) AS month,
    MAX(CASE WHEN T1.customer = 'yes' THEN T1.minutes ELSE -T1.minutes END) - MIN(CASE WHEN T1.customer = 'yes' THEN T1.minutes ELSE -T1.minutes END) AS diff
FROM 
    bike_usage_data AS T1
WHERE 
    YEAR(T1.date) = 2017
GROUP BY 
    MONTH(T1.date)
ORDER BY 
    diff DESC
LIMIT 1;
```