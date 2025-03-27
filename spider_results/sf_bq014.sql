Certainly! To achieve this, you need to join multiple tables and filter based on specific conditions. Here's an example SQL query that assumes you have the following tables:

1. `orders` table with columns:
   - `order_id`
   - `customer_id`
   - `product_category`
   - `status` (e.g., 'cancelled', 'returned')

2. `customers` table with columns:
   - `customer_id`
   - `first_non_cancelled_order_id`

3. `products` table with columns:
   - `product_category`
   - `revenue_per_unit`

Here's the SQL query to find the revenue for the product category with the highest number of customers making a purchase in their first non-cancelled and non-returned order:

```sql
SELECT 
    p.product_category,
    SUM(p.revenue_per_unit * o.quantity) AS total_revenue
FROM 
    orders o
JOIN 
    customers c ON o.customer_id = c.customer_id
JOIN 
    products p ON o.product_category = p.product_category
WHERE 
    o.status IN ('non-cancelled', 'non-returned')
GROUP BY 
    p.product_category
ORDER BY 
    COUNT(o.order_id) DESC
LIMIT 1;