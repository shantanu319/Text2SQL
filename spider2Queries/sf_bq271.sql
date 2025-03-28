```sql
SELECT 
    DATE_FORMAT(order_date, '%Y-%m') AS month,
    COUNT(DISTINCT order_id) AS number_of_orders,
    COUNT(DISTINCT purchaser_id) AS number_of_unique_purchasers,
    SUM(product_retail_price - product_cost) AS profit
FROM 
    orders
WHERE 
    YEAR(order_date) = 2021 AND
    YEAR(user_registration_date) = 2021 AND
    YEAR(inventory_item_creation_date) = 2021
GROUP BY 
    DATE_FORMAT(order_date, '%Y-%m'),
    user_country,
    product_department,
    product_category;
```

This SQL query generates a report that meets the specified requirements. It calculates the number of orders, unique purchasers, and profit for each month in 2021, filtering orders based on the registration date of the users and the creation date of the inventory items. The results are grouped by the users' country, product department, and product category. The `DATE_FORMAT` function is used to extract the month from the `order_date`.