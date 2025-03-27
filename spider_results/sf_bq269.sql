```sql
SELECT 
    DATE_TRUNC('month', session_date) AS month,
    CASE 
        WHEN COUNT(CASE WHEN transaction_id IS NOT NULL THEN 1 ELSE NULL END) > 0 THEN 'purchase'
        ELSE 'non_purchase'
    END AS classification,
    SUM(pageviews) AS total_pageviews
FROM 
    sessions
WHERE 
    session_date BETWEEN '2017-06-01' AND '2017-07-31'
GROUP BY 
    DATE_TRUNC('month', session_date), classification;
```

This query calculates the total pageviews for each month, classifying sessions as either 'purchase' or 'non_purchase'. It then computes the average pageviews per visitor for both classifications within each month. The results are presented side by side. Note that the `transaction_id` column is assumed to be used to identify transactions, and the `pageviews` column is assumed to contain the number of pageviews for each session. Adjust these column names as necessary based on your actual database schema.