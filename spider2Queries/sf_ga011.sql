```sql
SELECT DISTINCT cleaned_url AS page_path
FROM (
    SELECT 
        REPLACE(REPLACE(url, '//', '/'), '/', '') AS cleaned_url,
        SUM(page_views) AS total_page_views
    FROM 
        google_merchandise_store
    WHERE 
        url LIKE '%/store/%'
        AND DATE_FORMAT(date, '%Y-%m') = '2020-12'
    GROUP BY 
        cleaned_url
    ORDER BY 
        total_page_views DESC
    LIMIT 1 OFFSET 1
) AS subquery;
```