```sql
SELECT TOP 1 search_term
FROM weekly_searches
WHERE refresh_date = (
    SELECT MAX(refresh_date)
    FROM weekly_searches
) - INTERVAL '1 year'
ORDER BY rank DESC;
```