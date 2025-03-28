```sql
SELECT 
    liquor_category,
    AVG(per_liter_retail_price) AS avg_per_liter_retail_price_2021
FROM 
    sales_transactions
WHERE 
    state = 'Iowa' AND YEAR(sale_date) = 2021
GROUP BY 
    liquor_category
ORDER BY 
    avg_per_liter_retail_price_2021 DESC
LIMIT 10;

-- To get the average per-liter retail prices for the years 2019, 2020, and 2021:
SELECT 
    liquor_category,
    AVG(per_liter_retail_price) AS avg_per_liter_retail_price_2019,
    AVG(per_liter_retail_price) AS avg_per_liter_retail_price_2020,
    AVG(per_liter_retail_price) AS avg_per_liter_retail_price_2021
FROM 
    sales_transactions
WHERE 
    state = 'Iowa'
GROUP BY 
    liquor_category;
```

This SQL query first calculates the average per-liter retail price for each liquor category in Iowa in 2021. It then groups the results by liquor category and orders them in