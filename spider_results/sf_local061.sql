SELECT AVG(monthly_sales) FROM (
    SELECT 
        MONTHNAME(date) AS month,
        (CASE WHEN year = 2021 THEN (promo_total * (1 + (monthly_sales_2020 / monthly_sales_2019 - 1))) ELSE NULL END) AS projected_monthly_sales
    FROM (
        SELECT 
            DATE_FORMAT(date, '%Y-%m') AS date,
            YEAR(date) AS year,
            MONTH(date) AS month,
            SUM(promo_total) AS promo_total,
            SUM(monthly_sales) AS monthly_sales
        FROM products
        WHERE promo_total_id = 1 AND channel_total_id = 1
        GROUP BY DATE_FORMAT(date, '%Y-%m')
    ) AS subquery