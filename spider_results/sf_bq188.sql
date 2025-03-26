SELECT p.category_name, AVG(TIMESTAMPDIFF(MINUTE, p.view_time, e.event_time)) AS avg_time_spent_per_product_page_visit
FROM products p
JOIN events e ON p.product_id = e.product_id