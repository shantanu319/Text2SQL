```sql
SELECT 
    MONTH(first_purchase) AS month,
    COUNT(*) * 100 / (SELECT COUNT(*) FROM purchases) AS percentage_of_users
FROM 
    purchases
GROUP BY 
    MONTH(first_purchase)
ORDER BY 
    month;
```

This SQL query calculates the percentage of users who made a purchase in each of the first, second, third, and fourth months since their initial purchase. It uses the `MONTH` function to extract the month from the `first_purchase` date and groups the results by month. The `COUNT(*)` function is used to count the number of users who made a purchase in each month, and the percentage is calculated by dividing this count by the total number of users and multiplying by 100. Finally, the results are ordered by month.