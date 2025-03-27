```sql
SELECT 
    pc.category_name,
    SUM(p.quantity) AS total_quantity_purchased,
    AVG(TIMESTAMPDIFF(MINUTE, p.view_time, e.next_event_time)) AS avg_time_per_page_visit
FROM 
    products p
JOIN 
    product_categories pc ON p.category_id = pc.category_id
LEFT JOIN (
    SELECT 
        user_id,
        product_id,
        view_time,
        MIN(event_time) AS next_event_time
    FROM 
        events
    GROUP BY 
        user_id, product_id, view_time
) e ON p.user_id = e.user_id AND p.product_id = e.product_id
GROUP BY 
    pc.category_name
ORDER BY 
    total_quantity_purchased DESC
LIMIT 1;
```

This SQL query calculates the total quantity purchased for each product category and the average time spent per page visit for that category. It uses a subquery to find the next event time for each product page visit within the same session. The results are ordered by the total quantity purchased in descending order and limited to the top result.