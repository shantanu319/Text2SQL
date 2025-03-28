```sql
SELECT AVG(forecasted_annual_sales) AS average_forecasted_annual_sales
FROM (
    SELECT 
        product_id,
        SUM(sales * weight) / SUM(weight) AS forecasted_annual_sales
    FROM (
        SELECT 
            product_id,
            month,
            sales,
            CASE 
                WHEN (month - 1) % 12 = 6 THEN 1.5 -- Summer
                WHEN (month - 1) % 12 = 9 THEN 1.5 -- Fall
                ELSE 1.0
            END AS weight
        FROM (
            SELECT 
                product_id,
                DATE_TRUNC('month', date_column) AS month,
                sales
            FROM your_table_name
            WHERE product_id IN (4160, 7790)
              AND YEAR(date_column) BETWEEN 2016 AND 2018
              AND MONTH(date_column) BETWEEN 1 AND 12
        ) AS monthly_data
        JOIN (
            SELECT 
                product_id,
                TIME_STEP,
                seasonal_adjustment
            FROM your_seasonal_adjustments_table
            WHERE TIME_STEP >= 7 AND TIME_STEP <= 30