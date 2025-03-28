```sql
SELECT 
    CASE WHEN helmet = 'Yes' THEN 'Wearing Helmet' ELSE 'Not Wearing Helmet' END AS HelmetGroup,
    COUNT(CASE WHEN fatality = 'Yes' THEN 1 ELSE NULL END) / COUNT(*) * 100 AS FatalityPercentage
FROM 
    motorcycle_collisions
GROUP BY 
    Helmet;
```

This SQL query calculates the percentage of motorcyclist fatalities per collision for two groups: those who were wearing helmets and those who were not. It uses a `CASE` statement to categorize the collisions based on whether the rider was wearing a helmet or not, then counts the number of fatalities in each category and divides it by the total number of collisions in that category to get the percentage. The results are grouped by the helmet status.