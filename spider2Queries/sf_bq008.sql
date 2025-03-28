```sql
SELECT 
    CASE 
        WHEN T3.page = '/home' THEN T4.max_time
        ELSE NULL
    END AS max_home_time,
    T3.page AS next_page
FROM 
    visitor_campaigns AS T1
JOIN 
    visitor_accesses AS T2 ON T1.id = T2.visitor_id
JOIN 
    pages AS T3 ON T2.page_id = T3.id
LEFT JOIN (
    SELECT 
        visitor_id,
        MAX(time_spent) AS max_time
    FROM 
        page_visits
    WHERE 
        page = '/home'
    GROUP BY 
        visitor_id
) AS T4 ON T2.visitor_id = T4.visitor_id
WHERE 
    T1.campaign_name LIKE '%Data Share%'
    AND T2.page LIKE '/home%'
GROUP BY 
    T3.page;
```

This SQL query retrieves the page that visitors most commonly visited next after accessing the '/home' page, along with the maximum time they spent on the '/home' page before moving on. It uses a subquery to find the maximum time spent on the '/home' page for each visitor and joins it with the main query to filter and group the results based on the