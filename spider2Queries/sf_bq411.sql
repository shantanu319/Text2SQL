```sql
SELECT 
    refresh_date,
    term_1 AS google_trend_search_term_1,
    term_2 AS google_trend_search_term_2,
    term_3 AS google_trend_search_term_3
FROM 
    top_terms
WHERE 
    refresh_date BETWEEN '2024-09-01' AND '2024-09-14'
GROUP BY 
    refresh_date
ORDER BY 
    refresh_date DESC;
```