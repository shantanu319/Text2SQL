```sql
SELECT 
    YEAR(date) AS sighting_year,
    MIN(CASE WHEN date > '2023-01-01' THEN date END) AS earliest_date_after_january
FROM 
    sightings
WHERE 
    species = 'Sterna paradisaea'
    AND latitude > 40
GROUP BY 
    YEAR(date)
HAVING 
    COUNT(*) > 10
ORDER BY 
    earliest_date_after_january ASC
LIMIT 1;
```

This query selects the year with the earliest date after January when more than 10 sightings of Sterna paradisaea were recorded north of 40 degrees latitude. It groups the data by year, filters for the specified conditions, counts the sightings, and orders the results by the earliest date after January. The `LIMIT 1` clause ensures that only the year with the earliest such date is returned.