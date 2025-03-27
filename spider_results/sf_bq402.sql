```sql
SELECT 
    COUNT(DISTINCT fullVisitorId) AS unique_visitors,
    SUM(CASE WHEN totals.transactions IS NOT NULL THEN 1 ELSE 0 END) AS unique_purchasers,
    (SUM(CASE WHEN totals.transactions IS NOT NULL THEN 1 ELSE 0 END) / COUNT(DISTINCT fullVisitorId)) * 100 AS conversion_rate,
    AVG(totals.transactions) AS avg_transactions_per_purchaser
FROM 
    data-to-insights.ecommerce.web_analytics;
```