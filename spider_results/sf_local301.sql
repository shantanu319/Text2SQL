```sql
SELECT 
    YEAR(sales_date) AS year,
    (SUM(CASE WHEN WEEKDAY(sales_date) BETWEEN 3 AND 6 THEN sales_amount ELSE 0 END) - SUM(CASE WHEN WEEKDAY(sales_date) BETWEEN 0 AND 2 OR WEEKDAY(sales_date) = 4 THEN sales_amount ELSE 0 END)) / SUM(CASE WHEN WEEKDAY(sales_date) BETWEEN 0 AND 2 OR WEEKDAY(sales_date) = 4 THEN sales_amount ELSE 0 END) * 100 AS percentage_change
FROM 
    sales_data
WHERE 
    sales_date >= '2018-06-01' AND sales_date < '2018-07-01'
    OR sales_date >= '2019-06-01' AND sales_date < '2019-07-01'
    OR sales_date >= '2020-06-01' AND sales_date < '2020-07-01'
GROUP BY 
    YEAR(sales_date);
```

This SQL query calculates the percentage change in sales between the four weeks leading up to June 15