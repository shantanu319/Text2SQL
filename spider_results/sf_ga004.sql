```sql
SELECT AVG(pageviews_diff) AS avg_pageviews_diff
FROM (
    SELECT 
        CASE 
            WHEN purchase_event IS NOT NULL THEN 'purchaser'
            ELSE 'non-purchaser'
        END AS involvement,
        pageviews - LAG(pageviews, 1) OVER (PARTITION BY user_id ORDER BY date) AS pageviews_diff
    FROM 
        your_table_name
    WHERE 
        date BETWEEN '2020-12-01' AND '2020-12-31'
) AS subquery;
```

Replace `your_table_name` with the actual name of your table. This query calculates the average difference in pageviews between purchasers and non-purchasers for each user in December 2020. It uses a common table expression (CTE) to first determine if a user was involved in a purchase event and then calculates the difference in pageviews between consecutive days for each user. Finally, it averages these differences across all users.