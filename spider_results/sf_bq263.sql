SELECT MONTH(order_date) AS month,
       SUM(total_sales) AS total_sales,
       SUM(cost) AS total_cost,
       COUNT(*) AS num_complete_orders,
       (SUM(total_profit) / SUM(cost)) AS profit_to_cost_ratio