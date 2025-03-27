```sql
SELECT 
    YEAR,
    MINOR_CRIME_CATEGORY,
    ROUND((PERCENTAGE_SHARE / 100) * TOTAL_CRIMES, 2) AS ANNUAL_PERCENTAGE_SHARE
FROM 
    london_crime_data
WHERE 
    YEAR = 2008 AND CRIME_TYPE = 'Minor Crime'
ORDER BY 
    PERCENTAGE_SHARE DESC
LIMIT 5;
```