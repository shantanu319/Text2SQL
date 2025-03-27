```sql
SELECT AVG(T2.total_quantity) AS average_total_quantity
FROM final_packaging_combinations T1
JOIN (
    SELECT item_id, SUM(quantity) AS total_quantity
    FROM (
        SELECT item_id, quantity, 
               CASE WHEN parent_item_id IS NULL THEN 1 ELSE 0 END AS leaf_level
        FROM packaging_relationships
        UNION ALL
        SELECT item_id, quantity, 1 AS leaf_level
        FROM leaf_items
    ) T
    WHERE leaf_level = 1
    GROUP BY item_id
) T2 ON T1.combination_id = T2.item_id;
```