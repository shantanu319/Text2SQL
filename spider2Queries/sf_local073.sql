```sql
SELECT 
    ROW_ID,
    ORDER_ID,
    CUSTOMER_ID,
    PIZZA_NAME,
    CONCAT(
        PIZZA_NAME,
        ': ',
        GROUP_CONCAT(
            CASE 
                WHEN T3.Topping IN (SELECT Topping FROM Exclusions) THEN ''
                WHEN T3.Topping IN (SELECT Topping FROM Extras) THEN '2x'
                ELSE T3.Topping
            END ORDER BY T3.Topping ASC SEPARATOR ', '
        )
    ) AS Final_Ingredients
FROM 
    Pizza_Orders
JOIN 
    Pizza_Recipes ON Pizza_Orders.Pizza_ID = Pizza_Recipes.Pizza_ID
LEFT JOIN 
    Standard_Toppings AS T1 ON Pizza_Recipes.Standard_Toppings_ID = T1.Standard_Toppings_ID
LEFT JOIN 
    Exclusions AS T2 ON Pizza_Orders.Exclusions_ID = T2.Exclusions_ID
LEFT JOIN 
    Extras AS T3 ON Pizza_Orders.Extras_ID = T3.Extras_ID
GROUP BY 
    ROW_ID, ORDER_ID, PIZZA_NAME, ORDER_TIME
ORDER BY 
    ROW_ID;
```