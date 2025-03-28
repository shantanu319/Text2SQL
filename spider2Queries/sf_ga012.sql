```sql
SELECT 
    t1.transaction_id,
    SUM(t1.quantity) AS total_item_quantity,
    SUM(t1.purchase_revenue_usd) AS purchase_revenue_total_usd,
    SUM(t1.purchase_revenue_for_top_tax_rate_category_usd) AS purchase_revenue_for_top_tax_rate_category_usd
FROM 
    (
        SELECT 
            transaction_id,
            quantity,
            purchase_revenue_usd,
            purchase_revenue_for_top_tax_rate_category_usd,
            (tax_value_in_usd / purchase_revenue_in_usd) AS tax_rate
        FROM 
            purchase_events
        WHERE 
            event_date = '2020-11-30'
    ) t1
JOIN 
    (
        SELECT 
            item_category,
            MAX(tax_rate) AS max_tax_rate
        FROM 
            purchase_events
        WHERE 
            event_date = '2020-11-30'
        GROUP BY 
            item_category
    ) t2 ON t1.item_category = t2.item_category AND t1.tax_rate = t2.max_tax_rate
GROUP BY 
    t1.transaction_id;
```