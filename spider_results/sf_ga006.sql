```sql
SELECT 
    user_pseudo_id,
    AVG(ecommerce.purchase_revenue_in_usd) AS avg_purchase_revenue_per_session
FROM 
    your_table_name
WHERE 
    DATE(event_timestamp) BETWEEN '2020-11-01' AND '2020-11-30'
    AND event_name = 'purchase'
    AND ecommerce.purchase_revenue_in_usd IS NOT NULL
GROUP BY 
    user_pseudo_id, ga_session_id
HAVING 
    COUNT(DISTINCT ga_session_id) > 1;
```

Make sure to replace `your_table_name` with the actual name of your table. This query filters events within the specified date range, considers only those with an event name of 'purchase', and calculates the average purchase revenue per session for users who have made more than one purchase session. The `HAVING` clause ensures that only users with multiple sessions are included in the result.