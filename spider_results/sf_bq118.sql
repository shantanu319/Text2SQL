```sql
SELECT 
    age_group,
    AVG(deaths_from_discharge) AS avg_deaths_discharge,
    AVG(deaths_from_vehicle) AS avg_deaths_vehicle,
    (AVG(deaths_from_discharge) - AVG(deaths_from_vehicle)) / AVG(deaths_from_vehicle) * 100 AS percentage_increase
FROM 
    icd_10_codes
WHERE 
    description LIKE '%discharge%' AND description NOT IN ('Urethral discharge', 'Discharge of firework', 'Legal intervention involving firearm discharge')
GROUP BY 
    age_group;
```

This SQL query calculates the average number of deaths for individuals with ICD-10 codes containing "discharge" and those containing "vehicle," then computes the percentage increase in death rates between these two categories. The results are grouped by age group. Adjust the table and column names according to your actual database schema.