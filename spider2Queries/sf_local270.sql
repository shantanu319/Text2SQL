```sql
SELECT DISTINCT T1.name AS container_name, T2.name AS item_name
FROM packaging_container AS T1
JOIN item AS T2 ON T1.id = T2.container_id
WHERE T1.parent_id IS NULL AND (
    SELECT SUM(T3.quantity)
    FROM item AS T3
    WHERE T3.container_id IN (T1.id, T2.container_id)
) > 500;
```