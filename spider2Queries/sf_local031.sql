```sql
SELECT MAX(T.monthly_delivered_orders_volume) AS highest_monthly_volume
FROM (
    SELECT YEAR(order_date) AS year,
           MONTH(order_date) AS month,
           SUM(delivered_orders_volume) AS monthly_delivered_orders_volume
    FROM orders
    WHERE YEAR(order_date) IN (2016, 2017, 2018)
    GROUP BY YEAR(order_date), MONTH(order_date)
) T
WHERE T.year = (
    SELECT MIN(year)
    FROM (
        SELECT YEAR(order_date) AS year
        FROM orders
        WHERE YEAR(order_date) IN (2016, 2017, 2018)
        GROUP BY YEAR(order_date)
    ) T2
);
```