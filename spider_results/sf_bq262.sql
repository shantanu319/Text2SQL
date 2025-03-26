SELECT 
    YEAR(order_date) AS year,
    MONTH(order_date) AS month,
    product_category,
    SUM(CASE WHEN order_status = 'completed' THEN 1 ELSE 0 END) AS total_orders,
    SUM(price * quantity) AS total_revenue,
    SUM((price - cost) * quantity) AS total_profit,
    LAG(total_orders, 1) OVER (PARTITION BY product_category ORDER BY year, month) AS previous_month_total_orders,
    LAG(total_revenue, 1) OVER (PARTITION BY product_category ORDER BY year, month) AS previous_month_total_revenue,
    LAG(total_profit, 1) OVER (PARTITION BY product_category ORDER BY year, month) AS previous_month_total_profit,
    CASE 
        WHEN previous_month_total_orders IS NULL OR previous_month_total_revenue IS NULL OR previous_month_total_profit IS NULL THEN NULL
        ELSE ((total_orders - previous_month_total_orders) / previous_month_total_orders) * 100
    END AS month_over_month_growth_rate_order,
    CASE 
        WHEN previous_month_total_revenue IS NULL OR previous_month_total_profit IS NULL THEN NULL
        ELSE ((total_revenue - previous_month_total_revenue) / previous_month_total_revenue) * 100
    END AS month_over_month_growth_rate_revenue,
    CASE 
        WHEN previous_month_total_profit IS NULL THEN NULL
        ELSE ((total_profit - previous_month_total_profit) / previous_month_total_profit) * 100
    END AS month_over_month_growth_rate_profit
FROM 
    orders
WHERE 
    YEAR(order_date) >= 2019 AND MONTH(order_date) BETWEEN 6 AND 12
GROUP BY 
    YEAR(order_date), MONTH(order_date), product_category