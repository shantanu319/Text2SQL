To calculate the percentage of each country's population that had been confirmed with COVID-19 by June 30, 2020, you can use the following SQL query:

```sql
SELECT 
    wbc.CountryName,
    (wbc.Population * cdc.Confirmed / 1000000) AS PopulationPercentage
FROM 
    WorldBankData wbc
JOIN 
    COVID19OpenData cdc ON wbc.CountryCode = cdc.CountryCode
WHERE 
    cdc.Date = '2020-06-30'
ORDER BY 
    PopulationPercentage DESC;
```

### Explanation:
- **WorldBankData (wbc)**: This table contains population data for different countries.
- **COVID19OpenData (cdc)**: This table contains cumulative COVID-19 confirmed cases data for different countries.
- **JOIN**: We join these two tables on `CountryCode` to link population data with confirmed case data.
- **WHERE**: We filter the data to include only records from June 30, 2020.
- **Calculation**: For each country, we calculate the percentage of its population that had been confirmed with COVID-19 by dividing