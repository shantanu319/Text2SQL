```sql
WITH RankedProducts AS (
    SELECT 
        city,
        product_id,
        SUM(sales) AS total_sales,
        RANK() OVER (PARTITION BY city ORDER BY SUM(sales) DESC) AS product_rank
    FROM 
        sales_data
    WHERE 
        quarter IN ('Q4_2019', 'Q4_2020')
        AND promotion = FALSE
    GROUP BY 
        city, product_id
),
TopProducts AS (
    SELECT 
        city,
        product_id,
        total_sales,
        product_rank
    FROM 
        RankedProducts
    WHERE 
        product_rank <= 20
)
SELECT 
    tp.city,
    tp.product_id,
    tp.total_sales AS q4_2019_sales,
    t2.total_sales AS q4_2020_sales,
    (t2.total_sales - tp.total_sales) / tp.total_sales * 100 AS share_change
FROM 
    TopProducts tp
JOIN 
    sales_data t2 ON tp.city = t2.city AND tp.product_id = t2.product_id AND t2.quarter = 'Q4_2020'
ORDER BY 
    share_change DESC;
```