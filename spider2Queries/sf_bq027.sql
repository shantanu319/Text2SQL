```sql
SELECT 
    publication_number,
    COUNT(*) AS backward_citations_count
FROM 
    patents
WHERE 
    grant_date BETWEEN '2010-01-01' AND '2018-12-31'
    AND category = 'SEA'
GROUP BY 
    publication_number;
```