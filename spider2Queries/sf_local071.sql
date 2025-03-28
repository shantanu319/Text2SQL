```sql
SELECT T1.country_code, COUNT(*) AS streak_length
FROM (
    SELECT country_code, 
           ROW_NUMBER() OVER (PARTITION BY country_code ORDER BY date_inserted) - 
           ROW_NUMBER() OVER (PARTITION BY country_code ORDER BY date_inserted DESC) AS streak_number
    FROM your_table_name
    WHERE YEAR(date_inserted) = 2022 AND MONTH(date_inserted) = 6
) AS T1
WHERE T1.streak_number > 0
GROUP BY T1.country_code
ORDER BY streak_length DESC
LIMIT 2;
```

Replace `your_table_name` with the actual name of your table. This query calculates the streak length for each country by comparing the order of insertion dates within each country. It then filters out countries with a streak number greater than zero and orders the results by streak length in descending order, limiting the output to the top two countries. The 2-letter country codes are returned as required.