```sql
SELECT 
    COUNT(CASE WHEN injury_severity > 3 THEN 1 END) * 100 / COUNT(*) AS percentage_with_severe_injuries
FROM 
    accidents_2015
WHERE 
    accident_date BETWEEN '2015-01-01' AND '2015-08-31'
    AND num_involved > 1;
```

This SQL query calculates the percentage of accidents in the specified dataset where more than one individual has a severe injury. It filters the data based on the date range and the number of involved individuals, then counts the number of accidents with severe injuries and divides it by the total number of accidents to get the percentage. The result is multiplied by 100 to convert it to a percentage format.