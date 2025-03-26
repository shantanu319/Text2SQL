SELECT COUNT(*) AS num_orders, AVG(payment) AS avg_payment_per_order, (DATEDIFF(MAX(purchase_date), MIN(purchase_date)) / 7 + IF(DATEDIFF(MAX(purchase_date), MIN(purchase_date)) % 7 < 7, 0, 1)) AS customer_lifespan