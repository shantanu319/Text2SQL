SELECT 
    DATE_TRUNC('month', date) AS month,
    (SUM(add_to_cart_conversions) / SUM(pageviews)) * 100 AS add_to_cart_conversion_rate,
    (SUM(purchase_conversions) / SUM(pageviews)) * 100 AS purchase_conversion_rate
FROM 
    sales_data
WHERE 
    date BETWEEN '2017-01-01' AND '2017-03-31'
GROUP BY 
    DATE_TRUNC('month', date)
ORDER BY 
    month;  
```