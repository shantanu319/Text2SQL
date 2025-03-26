SELECT 
    COUNT(DISTINCT CASE WHEN totals.transactions IS NOT NULL THEN fullVisitorId END) AS num_purchasers,
    COUNT(DISTINCT fullVisitorId) AS num_visitors