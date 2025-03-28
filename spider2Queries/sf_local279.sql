To convert the given text into an SQL query, we need to break down the requirements and translate them into SQL syntax. Here's how you can structure the query:

```sql
WITH RECURSIVE MonthlyInventory AS (
    -- Base case: Starting from December 2018 inventory levels
    SELECT 
        product_id,
        '2018-12' AS month,
        ending_inventory,
        min_required_level,
        ABS(ending_inventory - min_required_level) AS abs_diff
    FROM 
        inventory_levels
    WHERE 
        year = 2018 AND month = 12

    UNION ALL

    -- Recursive case: Adjusting inventory for each subsequent month
    SELECT 
        i.product_id,
        DATE_FORMAT(DATE_ADD(m.month, INTERVAL 1 MONTH), '%Y-%m') AS month,
        i.ending_inventory,
        m.min_required_level,
        ABS(i.ending_inventory - m.min_required_level) AS abs_diff
    FROM 
        inventory_levels i
    JOIN 
        MonthlyInventory m ON i.year = m.year AND i.month = m.month + 1
)
SELECT 
    product_id,
    month,
    abs_diff
FROM 
    MonthlyInventory
WHERE