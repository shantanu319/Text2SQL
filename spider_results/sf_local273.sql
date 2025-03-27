To convert the given text into SQL, we need to break down the requirements and translate them into a query. Here's a step-by-step breakdown:

1. **Calculate the Average Pick Percentage**: This involves calculating the average of a pick percentage for each product name.

2. **First-In-First-Out (FIFO) Approach**: This means selecting inventory locations based on the earliest purchase date and smallest available quantity.

3. **Picked Quantity Calculation**: The picked quantity should reflect only the overlapping range between each order's required quantity and the inventory's available quantity.

4. **Grouping and Ordering**: Results should be grouped by product name and ordered by product name.

Given these requirements, here is the SQL query:

```sql
WITH Inventory AS (
    SELECT 
        product_name,
        location_id,
        purchase_date,
        available_quantity
    FROM 
        inventory_locations
),
OrderDetails AS (
    SELECT 
        order_id,
        product_name,
        required_quantity
    FROM 
        orders
),
PickQuantities AS (
    SELECT 
        od.order_id,
        od.product_name,
        od.required_quantity,
        il.available_quantity,
        CASE 
            WHEN il.available_quantity >= od.required_quantity THEN il.available_quantity
            ELSE od.required_quantity