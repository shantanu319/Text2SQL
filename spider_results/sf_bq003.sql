```sql
SELECT 
    DATE_TRUNC('month', session_date) AS month,
    CASE 
        WHEN COUNT(transactions) >= 1 AND product_revenue IS NOT NULL THEN 'purchase'
        ELSE 'non-purchase'
    END AS session_type,
    AVG(pageviews_per_visitor) AS avg_pageviews_per_visitor
FROM 
    hits_product_revenue
JOIN 
    totals_transactions ON hits_product_revenue.session_id = totals_transactions.session_id
WHERE 
    session_date BETWEEN '2017-04-01' AND '2017-07-31'
GROUP BY 
    DATE_TRUNC('month', session_date),
    session_type;
```