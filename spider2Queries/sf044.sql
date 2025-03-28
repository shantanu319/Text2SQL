```sql
SELECT 
    (SUM(CASE WHEN T2.post_market_close > T1.post_market_close THEN 1 ELSE 0 END) * 100.0 / COUNT(T2.post_market_close)) AS percentage_change
FROM 
    magnificent_7_companies AS T1
JOIN 
    stock_prices AS T2 ON T1.company_id = T2.company_id
WHERE 
    T2.date BETWEEN '2024-01-01' AND '2024-06-30';
```