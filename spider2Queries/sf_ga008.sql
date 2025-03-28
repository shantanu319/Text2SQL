```sql
SELECT 
    DATE(log_date) AS log_day,
    COUNT(DISTINCT page_view_id) AS total_page_views,
    AVG(page_views_per_user) AS avg_page_views_per_user
FROM 
    logs
WHERE 
    log_date BETWEEN '2020-11-01' AND '2020-11-30'
GROUP BY 
    DATE(log_date)
HAVING 
    COUNT(page_view_id) > 0;
```

This SQL query calculates the total number of page views and the average number of page views per user for each day in November 2020. It filters out days where no page views were recorded by using a `HAVING` clause with a condition that ensures there is at least one page view on each day. The results are grouped by the date of the log entry.