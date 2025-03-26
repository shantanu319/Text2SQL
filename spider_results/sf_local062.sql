SELECT COUNT(*) AS num_customers, MIN(total_profit) AS min_total_profit, MAX(total_profit) AS max_total_profit FROM (
    SELECT SUM(quantity_sold * (unit_price - unit_cost)) AS total_profit, 
           FLOOR((SUM(profit) / 1000000) % 10) AS bucket_id
    FROM customer