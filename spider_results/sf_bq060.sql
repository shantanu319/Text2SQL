```sql
SELECT 
    country,
    SUM(net_migration) AS total_net_migration
FROM 
    migration_data
WHERE 
    area > 500 AND year = 2017
GROUP BY 
    country
ORDER BY 
    total_net_migration DESC
LIMIT 3;
```

This query calculates the total net migration for each country with an area greater than 500 square kilometers in 2017, orders them by the total net migration in descending order, and limits the result to the top 3 countries. It does not include migration rates directly; if you need that information, you would need additional data or a different approach. If you have specific requirements for migration rates, please provide more details.