```sql
SELECT 
    Borough,
    COUNT(*) AS Total_Stations,
    SUM(CASE WHEN Actual_Entry = 'Yes' AND ADA_Compliant = 'Yes' THEN 1 ELSE 0 END) AS Stations_with_Both,
    (SUM(CASE WHEN Actual_Entry = 'Yes' AND ADA_Compliant = 'Yes' THEN 1 ELSE 0 END) * 1.0 / COUNT(*)) * 100 AS Percentage_of_Total
FROM 
    Subway_Stations
WHERE 
    Borough IN ('Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island')
GROUP BY 
    Borough
ORDER BY 
    Percentage_of_Total DESC;
```

This SQL query calculates the total number of subway stations in each New York City borough, counts how many stations have both an actual entry and are ADA-compliant, and then computes the percentage of these stations relative to the total number of stations in each borough. The results are ordered by the percentage in descending order. Make sure to replace `Subway_Stations` with the actual table name used in your database. Also, ensure that the column names (`Borough`, `Actual_Entry`, `ADA_Compliant`) match those in your database schema. <