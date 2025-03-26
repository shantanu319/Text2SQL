SELECT TOP 20 p.product_name, q4_2019_sales_share, q4_2020_sales_share, q4_2020_sales_share - q4_2019_sales_share AS share_change FROM (
    SELECT 
        p.product_name,
        SUM(CASE WHEN promotion = 'No' THEN s.sales ELSE 0 END) / SUM(s.sales) * 100 AS q4_2019_sales_share,
        SUM(CASE WHEN promotion = 'No' THEN s.sales ELSE 0 END) / SUM(s.sales) * 100 AS q4_2020_sales_share
    FROM 
        products p
    JOIN 
        sales s ON p.id = s.product_id
    WHERE 
        s.quarter = 'Q4 2019'
    GROUP BY 
        p.product_name
)