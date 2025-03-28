```sql
SELECT item_name, SUM(sales_revenue) AS total_sales_revenue
FROM sales_data
WHERE YEAR = 2023
GROUP BY item_name
ORDER BY (SUM(sales_revenue) - LAG(SUM(sales_revenue)) OVER (PARTITION BY item_name ORDER BY YEAR)) / LAG(SUM(sales_revenue)) OVER (PARTITION BY item_name ORDER BY YEAR) DESC
LIMIT 5;
```

This query calculates the year-over-year growth percentage for each item by comparing the current year's sales revenue to the previous year's. It then orders the results by this growth percentage in descending order and limits the output to the top 5 items. Note that this assumes a table named `sales_data` with columns `item_name`, `sales_revenue`, and `YEAR`. Adjust the column names as necessary to fit your database schema. Additionally, if you want to handle cases where there is no previous year's data, you might need to add additional logic or adjust the window function accordingly.