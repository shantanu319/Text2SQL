```sql
SELECT 
    DATE,
    UPPER(CITY) AS CITY_NAME
FROM 
    your_table_name
WHERE 
    YEAR(DATE) = 2021 AND MONTH(DATE) = 7
ORDER BY 
    DATE;
```

This query selects the `DATE` and `CITY` columns from your table, filters for records where the year is 2021 and the month is July, orders the results by date, and capitalizes the first letter of each city name using the `UPPER()` function. Adjust `your_table_name` to match the actual name of your table.