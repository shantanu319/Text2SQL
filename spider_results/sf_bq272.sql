SELECT p.name AS Product_Name, m.month_name AS Month_Name, SUM(o.sale_price - o.cost) AS Profit
FROM Products p
JOIN Orders o ON p.id = o.product_id
WHERE o.status IN ('Shipped', 'Delivered')
GROUP BY p.name, m.month_name