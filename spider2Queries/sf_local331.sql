```sql
SELECT 
    t3.page AS third_page,
    COUNT(*) AS frequency
FROM 
    (
        SELECT 
            user_id,
            page,
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp) AS visit_order
        FROM 
            visits
        WHERE 
            page = '/detail'
    ) t1
JOIN 
    (
        SELECT 
            user_id,
            page,
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp) AS visit_order
        FROM 
            visits
        WHERE 
            page = '/detail'
    ) t2 ON t1.user_id = t2.user_id AND t1.visit_order + 1 = t2.visit_order
JOIN 
    (
        SELECT 
            user_id,
            page,
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp) AS visit_order
        FROM 
            visits
        WHERE 
            page = '/detail'
    ) t3 ON t2.user_id = t3.user_id AND t2.visit_order + 1 = t3.visit_order
WHERE 
    t1.visit_order = 1 AND t2.visit_order = 2 AND t3.visit_order = 3
GROUP BY 
    t