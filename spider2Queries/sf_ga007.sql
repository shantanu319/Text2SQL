```sql
SELECT 
    (COUNT(CASE WHEN page_type = 'PDP' THEN 1 ELSE NULL END) * 100.0 / COUNT(*)) AS pdp_percentage
FROM 
    page_views
WHERE 
    date = '2021-01-02';
```