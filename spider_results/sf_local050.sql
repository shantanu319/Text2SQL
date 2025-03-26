SELECT AVG(monthly_sales) FROM (
    SELECT 
        (monthly_sales * (promo_growth_rate_2020 / promo_growth_rate_2019)) AS monthly_sales
    FROM 
        (