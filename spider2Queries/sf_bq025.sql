```sql
SELECT 
    Country,
    TotalPopulationUnder20,
    MidYearPopulation,
    (TotalPopulationUnder20 / MidYearPopulation) * 100 AS PercentageUnder20
FROM 
    PopulationData
WHERE 
    Year = 2020
ORDER BY 
    PercentageUnder20 DESC
LIMIT 10;
```