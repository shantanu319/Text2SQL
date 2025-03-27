```sql
SELECT 
    (SUM(CASE WHEN YEAR(T1.date) = 2023 THEN T1.gross_income ELSE 0 END) - SUM(CASE WHEN YEAR(T1.date) = 2022 THEN T1.gross_income ELSE 0 END)) / SUM(CASE WHEN YEAR(T1.date) = 2022 THEN T1.gross_income ELSE 0 END) * 100 AS percentage_change_gross_income,
    AVG(T2.purchase_only_home_price_index) AS seasonally_adjusted_purchase_only_home_price_index
FROM 
    financial_entities T1
JOIN 
    metro_areas T2 ON T1.metro_area_id = T2.id
WHERE 
    T1.metro_area_name = 'Phoenix-Mesa-Scottsdale, AZ' AND T1.date BETWEEN '2023-01-01' AND '2023-12-31'
GROUP BY 
    T1.metro_area_name;
```

This SQL query calculates the percentage change in gross income inflow and the seasonally-adjusted purchase-only home price index for the specified period and metro area. It joins the `financial_entities` table with the `metro_areas` table on their respective