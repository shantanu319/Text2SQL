SELECT hub_name FROM orders WHERE (finished_orders - previous_month_finished_orders) / previous_month_finished_orders > 0.2