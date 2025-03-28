```sql
WITH MonthlyOrders AS (
    SELECT 
        o.order_id,
        p.product_category,
        DATE_TRUNC('month', o.delivery_date) AS delivery_month,
        SUM(o.sale_price) AS total_revenue,
        COUNT(*) AS total_orders
    FROM 
        orders o
    JOIN 
        products p ON o.product_id = p.product_id
    WHERE 
        o.status = 'Complete'
        AND o.delivery_date < '2022-01-01'
    GROUP BY 
        o.order_id, p.product_category, delivery_month
),
MonthlyGrowth AS (
    SELECT 
        mo.product_category,
        mo.delivery_month,
        mo.total_revenue,
        mo.total_orders,
        LAG(mo.total_revenue) OVER (PARTITION BY mo.product_category ORDER BY mo.delivery_month) AS prev_total_revenue,
        LAG(mo.total_orders) OVER (PARTITION BY mo.product_category ORDER BY mo.delivery_month) AS prev_total_orders,
        (mo.total_revenue - prev_total_revenue) / NULLIF(prev_total_revenue, 0) * 100 AS revenue_growth,
        (mo.total_orders - prev_total_orders) / NULLIF(prev_total_orders, 0