```sql
SELECT DISTINCT ProductName
FROM Purchases
WHERE PurchaseDate BETWEEN '2017-07-01' AND '2017-07-31'
  AND ProductName NOT LIKE '%YouTube%'
GROUP BY ProductName
ORDER BY SUM(Quantity) DESC
LIMIT 1;
```