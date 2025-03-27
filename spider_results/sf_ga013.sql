```sql
SELECT 
    page_name,
    CASE 
        WHEN page_name LIKE '%PDP%' THEN 'PDP'
        WHEN page_name LIKE '%PLP%' THEN 'PLP'
        ELSE page_name
    END AS adjusted_page_name
FROM 
    visits
WHERE 
    user_id = '1402138.5184246691' AND
    visit_date = '2021-01-02';
```