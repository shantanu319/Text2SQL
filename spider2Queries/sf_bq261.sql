To convert the given text into an SQL query, we need to break down the requirements and construct a query that processes the data accordingly. Here's a step-by-step breakdown:

1. **Identify the relevant tables**: We assume there are at least two tables involved:
   - `orders`: Contains information about each order, including the order date.
   - `order_items`: Contains details about each item in an order, including the sale price and product cost.

2. **Calculate the total profit for each product per month**: We need to calculate the total profit for each product per month by subtracting the product cost from the sale price.

3. **Find the product with the highest total profit per month**: For each month, we need to find the product that achieved the highest total profit.

4. **Report the total cost and total profit for that top product per month**: We need to include all order items for the top product per month, regardless of their status.

5. **Present the results chronologically by month**: The results should be ordered by month.

Here's the SQL query based on these steps:

```sql
WITH MonthlyProfit AS (
    SELECT 
        o.order_date,
        oi.product_id,
        SUM(oi.sale_price - oi.product_cost