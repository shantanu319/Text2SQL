```sql
SELECT SUM(TotalPrice) AS TotalIncome
FROM (
    SELECT 
        CASE 
            WHEN T3.PizzaName = 'Meat Lovers' THEN 12 + (T4.ExtraToppings * 1)
            WHEN T3.PizzaName = 'Vegetarian' THEN 10 + (T4.ExtraToppings * 1)
            ELSE 0
        END AS TotalPrice,
        T1.OrderID
    FROM Orders AS T1
    INNER JOIN OrderItems AS T2 ON T1.OrderID = T2.OrderID
    INNER JOIN Pizzas AS T3 ON T2.PizzaID = T3.PizzaID
    LEFT JOIN ExtraToppings AS T4 ON T2.ExtraToppingID = T4.ExtraToppingID
    WHERE T1.Status = 'Completed'
) AS SubQuery;
```