Certainly! To generate a report showing the number of delivered orders for each month in the years 2016, 2017, and 2018, you can use the following SQL query. This assumes you have a table named `orders` with columns `order_id`, `delivery_date`, and `year`.

```sql
SELECT 
    YEAR(delivery_date) AS Year,
    MONTH(delivery_date) AS Month,
    COUNT(order_id) AS Number_of_Delivered_Orders
FROM 
    orders
WHERE 
    YEAR(delivery_date) IN (2016, 2017, 2018)
GROUP BY 
    YEAR(delivery_date), MONTH(delivery_date)
ORDER BY 
    Year, Month;
```

This query will return a result set where each row represents a month and year combination, along with the count of delivered orders for that specific period. The results are grouped by year and month, and then ordered by year and month for clarity. Adjust the table and column names as necessary to fit your database schema.